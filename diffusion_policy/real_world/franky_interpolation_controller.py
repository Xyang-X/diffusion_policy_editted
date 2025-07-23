import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R
import numpy as np
from franky import RelativeDynamicsFactor,Robot, CartesianMotion, JointMotion, Affine, Twist, CartesianVelocityMotion, JointMotion
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    VELOCITY = 3
    RESET_POSE = 4


class FR3InterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip, 
            frequency=125, 
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            pos_speed_scale=1.0, # scale the max_pos_speed by this factor
            rot_speed_scale=1.0, # scale the max_rot_speed by this factor
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=0.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,
            home_pose=None
            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        assert pos_speed_scale > 0
        assert rot_speed_scale > 0
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FR3Controller")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.pos_speed_scale = pos_speed_scale
        self.rot_speed_scale = rot_speed_scale
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.home_pose = home_pose



        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
            'linear_velocity': np.zeros(3, dtype=np.float32),  
            'angular_velocity': np.zeros(3, dtype=np.float32), 
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [ 
            'current_cartesian_state.pose.end_effector_pose',
            'current_cartesian_state.velocity.end_effector_twist',
            'current_joint_state.position',
            'current_joint_state.velocity'
        ]
        robot=Robot(robot_ip)
        
        example = dict()
        example['ActualTCPPose'] = np.array(getattr(robot.current_cartesian_state.pose, 'end_effector_pose'))
        example['ActualTCPSpeed'] = np.array(getattr(robot.current_cartesian_state.velocity, 'end_effector_twist'))
        example['ActualQ'] = np.array(robot.current_joint_state.position)
        example['ActualQd'] = np.array(robot.current_joint_state.velocity)
        # for key in receive_keys:
        #     example[key] = np.array(self.get_nested_attr(robot, key))
        example['robot_receive_timestamp'] = time.time()
        example= {**example, **self.FR3_state_transform(dict(list(example.items())[:2]))}
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
        self.back_to_home = False

    # ========= assistant method ===========
    # get and transform FR3 data into numpy array 
    def get_nested_attr(self, obj, attr_path):
        """支持 'a.b.c' 形式的属性路径"""
        for attr in attr_path.split('.'):
            obj = getattr(obj, attr)
        return obj
        
    def FR3_state_transform(self, state):
        """
        Transform FR3 state to numpy array.
        """
        keys=['ActualTCPPose',
              'ActualTCPSpeed']
        transformed_state = dict()

        transformed_state[keys[0]] = self.Affine_to_pose(state[keys[0]].item())

        transformed_state[keys[1]] = np.append(state[keys[1]].item().linear, state[keys[1]].item().angular)
        
        return transformed_state
    
    def Affine_to_pose(self, affine):
        """
        Convert an Affine object to a 6D pose (3D position + quaternion).
        """
        r = R.from_quat(affine.quaternion)
        rvec = r.as_rotvec()
        return np.append(affine.translation, rvec)

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        print("[RTDEPositionalController] Stop command sent to controller process.")
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        # print('message:',message)
        self.input_queue.put(message)
        
    def velocity(self, linear_velocity, angular_velocity):
        """
        linear_velocity: 3d vector
        angular_velocity: 3d vector
        """
        assert self.is_alive()
        linear_velocity = np.array(linear_velocity)
        angular_velocity = np.array(angular_velocity)
        assert linear_velocity.shape == (3,)
        assert angular_velocity.shape == (3,)

        message = {
            'cmd': Command.VELOCITY.value,
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
        }
        self.input_queue.put(message)

    def reset_pose(self):
        """
        Reset the robot pose to initial joint angles.
        """
        # upper_limit = np.array([166,105,-7,165,265,175])/ 180 * np.pi
        # lower_limit = np.array([-166,-105,-176,-165, 25,-175])/ 180 * np.pi
        
        # assert np.all((self.joints_init <= upper_limit) & (self.joints_init >= lower_limit)), \
        #     f"Joints init {self.joints_init} out of bounds, should be in [{lower_limit}, {upper_limit}]"

        # convert home pose to Affine
        if not self.back_to_home:
            message = {
                'cmd': Command.RESET_POSE.value,
            }
            self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start rtde
        robot_ip = self.robot_ip
        robot=Robot(robot_ip)


        try:
            if self.verbose:
                print(f"[RTDEPositionalController] Connect to robot: {robot_ip}")

            # # set parameters
            # if self.tcp_offset_pose is not None:
            #     rtde_c.setTcp(self.tcp_offset_pose)
            # if self.payload_mass is not None:
            #     if self.payload_cog is not None:
            #         assert rtde_c.setPayload(self.payload_mass, self.payload_cog)
            #     else:
            #         assert rtde_c.setPayload(self.payload_mass)
            
            # init pose
            print('Joint init: ',self.joints_init is not None)
            if self.joints_init is not None:
                # upper_limit = np.array([166,105,-7,165,265,175])/ 180 * np.pi
                # lower_limit = np.array([-166,-105,-176,-165, 25,-175])/ 180 * np.pi
                
                # assert np.all((self.joints_init <= upper_limit) & (self.joints_init >= lower_limit)), \
                #     f"Joints init {self.joints_init} out of bounds, should be in [{lower_limit}, {upper_limit}]"
                rdf=RelativeDynamicsFactor(0.1,0.2,0.1)
                print('Inintializing')
                m_ji= JointMotion(self.joints_init,relative_dynamics_factor=rdf)
                robot.move(m_ji)

            # main loop
            dt = 1. / self.frequency
            curr_pose = self.Affine_to_pose(robot.current_cartesian_state.pose.end_effector_pose)
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )
            
            iter_idx = 0
            keep_running = True
            twist = None
            vel_ctl = 0
            self.back_to_home = False
            while keep_running:
                t1=time.time()
                # start control iteration
                # t_start = rtde_c.initPeriod()
                if self.back_to_home and self.home_pose is not None:
                    time.sleep(3)  # wait for robot to stop
                    # home_affine = Affine(
                    #     translation=self.home_pose[:3],
                    #     quaternion=R.from_rotvec(self.home_pose[3:]).as_quat()
                    # )
                    # m_bh = CartesianMotion(home_affine)
                    m_bh = JointMotion(self.joints_init,relative_dynamics_factor=RelativeDynamicsFactor(0.3,0.3,0.1))
                    robot.move(m_bh)
                    print("Franka Moved back to home pose.")
                    self.back_to_home = False
                # send command to robot
                # t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                # pose_command = pose_interp(t_now)
                vel = 0.5
                acc = 0.2
                jerk = 0.1
                rdf = RelativeDynamicsFactor(vel, acc, jerk)
                # assert rtde_c.servoL(pose_command, 
                #     vel, acc, # dummy, not used by ur5
                #     dt, 
                #     self.lookahead_time, 
                #     self.gain)
                if not vel_ctl:
                    # quat = st.Rotation.from_rotvec(pose_command[3:]).as_quat()
                    
                    # m_cv = CartesianMotion(Affine(pose_command[:3], quat),relative_dynamics_factor=rdf)
                    # print("[PositionalController] Move to pose:"                                                                                        )
                    # print(pose_command)
                
                    m_cv=CartesianVelocityMotion(Twist(np.zeros(3),np.zeros(3)),relative_dynamics_factor=rdf)
                    # print('None command')
                else:
                    m_cv = CartesianVelocityMotion(twist,relative_dynamics_factor=0.2)
                    # print("[PositionalController] Move with velocity:")
                    # print('linear:', twist.linear, 'angular:', twist.angular)


                robot.move(m_cv,asynchronous=True)
                # print('Command velocity:',m_cv.linear)


                # update robot state
                state = dict()
                state['ActualTCPPose'] = np.array(getattr(robot.current_cartesian_state.pose, 'end_effector_pose'))
                state['ActualTCPSpeed'] = np.array(getattr(robot.current_cartesian_state.velocity, 'end_effector_twist'))
                state['ActualQ'] = np.array(robot.current_joint_state.position)
                state['ActualQd'] = np.array(robot.current_joint_state.velocity)
                # for key in self.receive_keys:
                #     state[key] = np.array(self.get_nested_attr(robot, key))
                state = {**state, **self.FR3_state_transform(dict(list(state.items())[:2]))}
                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.VELOCITY.value:
                        vel_ctl=1
                        linear_velocity = command['linear_velocity']
                        angular_velocity = command['angular_velocity']
                        twist = Twist(linear_velocity*self.pos_speed_scale, angular_velocity*self.rot_speed_scale)
                    elif cmd == Command.RESET_POSE.value:
                        self.back_to_home = True
                    else:
                        keep_running = False
                        break

                # regulate frequency
                # rtde_c.waitPeriod(t_start)
                time.sleep(0.001)
                t2=time.time()
                # print(f'Control period: {t2-t1:.4f}s, Commands: {twist}')
                

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # if self.verbose:
                #     print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")
        except Exception as e:
            print("[Controller] Exception occurred in robot process:")
            import traceback
            traceback.print_exc()

        finally:
            # manditory cleanup
            # decelerate
            # rtde_c.servoStop()

            # # terminate
            # rtde_c.stopScript()
            # rtde_c.disconnect()
            # rtde_r.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {robot_ip}")
