import os
import time
import enum
import numpy as np
import cv2
import multiprocessing as mp
from threading import Event
from multiprocessing.managers import SharedMemoryManager
from threadpoolctl import threadpool_limits
from typing import Optional, Callable, Dict
import sys
sys.path.append('/home/xy/DP/diffusion_policy')

from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from diffusion_policy.real_world.video_recorder import VideoRecorder

class Command(enum.Enum):
    # STOP = 0
    # SET_COLOR_OPTION = 3
    # START_RECORDING = 1
    # STOP_RECORDING = 2
    STOP = 0
    START_RECORDING = 1
    STOP_RECORDING = 2

class SingleCamera(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 serial_number=4,
                 resolution=(1280, 720),
                 capture_fps=30,
                 put_fps=None,
                 record_fps=None,
                 put_downsample=False,
                 transform=None,
                 vis_transform=None,
                 recording_transform=None,
                 video_recorder: Optional[VideoRecorder] = None,
                 verbose=False,
                 enable_color=True,
                 is_fisheye=False):
        super().__init__(name=f"SingleCamera-{serial_number}")
        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        if video_recorder is None:
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps,
                codec='h264',
                input_pix_fmt='bgr24',   # OpenCV 图像格式
                crf=18,
                thread_type='FRAME',
                thread_count=1
            )
        
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder 
        self.verbose = verbose
        self.enable_color = enable_color

        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.put_start_time = None

        self.is_fisheye = is_fisheye
        if self.is_fisheye:
            self.resolution = (640, 480)  # Fisheye camera resolution
            K = np.load('diffusion_policy/real_world/camera_mat.npy')
            D = np.load('diffusion_policy/real_world/distortion_coe.npy')
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.identity(3), K, self.resolution, cv2.CV_16SC2)
            self.map=[map1, map2]

        example = dict()
        if self.enable_color:
            example['color'] = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        example['timestamp'] = 0.0
        example['camera_capture_timestamp'] = 0.0
        example['camera_receive_timestamp'] = 0.0
        example['step_idx'] = 0
        example['video_path'] = np.array('a'*self.MAX_PATH_LENGTH)  # for video recording command
        example['recording_start_time']= 0.0
        # example['cmd'] = Command.SET_COLOR_OPTION.value  # default command
        example['cmd'] = 0

        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=128,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        self.vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=32,
            get_time_budget=0.2,
            put_desired_frequency=5
        )

        self.command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=8
        )

    def run(self):
        threadpool_limits(1)
        cv2.setNumThreads(1)

        cap = cv2.VideoCapture(int(self.serial_number))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.capture_fps)

        if not cap.isOpened():
            print(f"[Camera {self.serial_number}] Failed to open camera.")
            self.ready_event.set()
            return

        iter_idx = 0
        put_idx = None
        put_start_time = self.put_start_time or time.time()

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            receive_time = time.time()
            if not ret:
                continue

            if self.is_fisheye:
                frame = cv2.remap(frame, self.map[0], self.map[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                # print('Fisheye camera image size:', frame.shape)
            data = dict()
            data['camera_receive_timestamp'] = receive_time
            data['camera_capture_timestamp'] = receive_time
            if self.enable_color:
                data['color'] = frame


            put_data = self.transform(data) if self.transform else data
            # print('Camera '+str(self.serial_number)+' put_data: ',put_data['color'].shape)
            if self.put_downsample:
                local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                    timestamps=[receive_time],
                    start_time=put_start_time,
                    dt=1 / self.put_fps,
                    next_global_idx=put_idx,
                    allow_negative=True
                )
                for step_idx in global_idxs:
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)
            else:
                step_idx = int((receive_time - put_start_time) * self.put_fps)
                put_data['step_idx'] = step_idx
                put_data['timestamp'] = receive_time
                self.ring_buffer.put(put_data, wait=False)

            if iter_idx == 0:
                self.ready_event.set()
            # vis_data = self.vis_transform(data) if self.vis_transform and not self.is_fisheye else put_data

            # self.vis_ring_buffer.put(vis_data, wait=False)
            # print('Camera '+str(self.serial_number)+' vis shape:',vis_data['color'].shape)
            self.vis_ring_buffer.put(data, wait=True)


            rec_data = self.recording_transform(data) if self.recording_transform else put_data
            if self.video_recorder.is_ready():
                self.video_recorder.write_frame(rec_data['color'], frame_time=receive_time)

            try:
                commands = self.command_queue.get_all()
                for i in range(len(commands['cmd'])):
                    cmd = commands['cmd'][i]
                    if cmd == Command.STOP.value:
                        self.stop_event.set()
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(commands['video_path'][0])
                        start_time = commands['recording_start_time']
                        print('video_path:', video_path)
                        self.video_recorder.start(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop()
            except Empty:
                pass

            iter_idx += 1

        self.video_recorder.stop()
        cap.release()
        self.ready_event.set()
        print(f'[SingleCamera {self.serial_number}] Exiting worker process.')

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait(timeout=5.0)

    def stop(self, wait=True):
        self.command_queue.put({'cmd': Command.STOP.value})
        if wait:
            self.join()

    # def start_recording(self):
    #     self.command_queue.put({'cmd': Command.START_RECORDING.value})
    def start_recording(self, video_path: str, start_time: float=-1):
        assert self.enable_color

        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })

    def stop_recording(self):
        self.command_queue.put({'cmd': Command.STOP_RECORDING.value})

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    @staticmethod
    def get_available_camera_ids(max_id=10):
        available = []
        for i in range(max_id):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
    
    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

if __name__ == '__main__':
    from multiprocessing import set_start_method
    set_start_method('spawn')
    from multiprocessing.managers import SharedMemoryManager

    with SharedMemoryManager() as shm:
        cam = SingleCamera(
            shm_manager=shm,
            serial_number=4,  # OpenCV 摄像头编号
            capture_fps=30,
            verbose=True
        )
        cam.start()
        # cam.start_wait()
        print("Camera ready.")
        time.sleep(10)
        cam.stop()

