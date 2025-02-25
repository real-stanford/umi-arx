"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import sys
import os
from queue import Queue
from omegaconf import open_dict

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from omegaconf import OmegaConf
from utils.other_util import precise_wait
from peripherals.keystroke_counter import KeystrokeCounter, Key, KeyCode
from utils.real_inference_util import (
    get_real_obs_dict,
    get_real_obs_resolution,
    get_real_umi_obs_dict,
    get_real_umi_action,
)
from utils.cv_util import draw_predefined_mask
from peripherals.spacemouse_shared_memory import Spacemouse
from utils.pose_util import pose_to_mat, mat_to_pose, rot6d_to_mat
import scipy.spatial.transform as st
from modules.arx5_env import Arx5Env
import zmq

OmegaConf.register_new_resolver("eval", eval, replace=True)


def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = (
        np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    )
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta


def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0])  # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(
                robots_config[this_robot_idx]["sphere_center"]
            )
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(
                robots_config[that_robot_idx]["sphere_center"]
            )
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = (
                robots_config[this_robot_idx]["sphere_radius"]
                + robots_config[that_robot_idx]["sphere_radius"]
            )
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print("avoid collision between two arms")
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal

                ee_poses[this_robot_idx][:6] = mat_to_pose(
                    this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local)
                )
                ee_poses[that_robot_idx][:6] = mat_to_pose(
                    np.linalg.inv(this_that_mat)
                    @ that_sphere_mat_global
                    @ np.linalg.inv(that_sphere_mat_local)
                )


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--output", "-o", required=True, help="Directory to save recording")
@click.option("--policy_ip", default="localhost")
@click.option("--policy_port", default=8766)
@click.option("--robot_port", default=8765)
@click.option("--match_dataset_path", default=None, type=str)
@click.option(
    "--match_episode",
    default=None,
    type=int,
    help="Match specific episode from the match dataset",
)
@click.option("--camera_reorder", "-cr", default="0")
@click.option(
    "--vis_camera_idx", default=0, type=int, help="Which RealSense camera to visualize."
)
@click.option(
    "--init_joints",
    "-j",
    is_flag=True,
    default=False,
    help="Whether to initialize robot joint configuration in the beginning.",
)
@click.option(
    "--steps_per_inference",
    "-si",
    default=4,
    type=int,
    help="Action horizon for inference.",
)
@click.option(
    "--max_duration",
    "-md",
    default=2000000,
    help="Max duration for each epoch in seconds.",
)
@click.option(
    "--frequency", "-f", default=5, type=float, help="Control frequency in Hz."
)
@click.option(
    "--command_latency",
    "-cl",
    default=0.01,
    type=float,
    help="Latency between receiving SapceMouse command to executing on Robot in Sec.",
)
@click.option("-nm", "--no_mirror", is_flag=True, default=False)
@click.option("-sf", "--sim_fov", type=float, default=None)
@click.option("-ci", "--camera_intrinsics", type=str, default=None)
@click.option("--mirror_swap", is_flag=True, default=False)
@click.option("--task_name", type=str, default='')
def main(
    input,
    output,
    policy_ip,
    policy_port,
    robot_port,
    match_dataset_path,
    match_episode,
    camera_reorder,
    vis_camera_idx,
    init_joints,
    steps_per_inference,
    max_duration,
    frequency,
    command_latency,
    no_mirror,
    sim_fov,
    camera_intrinsics,
    mirror_swap,
    task_name,
):
    pid = os.getpid()
    os.sched_setaffinity(pid, [7])
    max_gripper_width = 0.085
    gripper_speed = 0.04
    cartesian_speed = 0.4
    orientation_speed = 0.8
    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, "obs"), exist_ok=True)
    os.makedirs(os.path.join(output, "action"), exist_ok=True)

    if match_dataset_path is not None:
        assert match_episode is not None
        assert os.path.exists(os.path.join(match_dataset_path, "obs", f"{match_episode}/0.npy")), "match_episode does not exist"
        match_episode_min = match_episode
        while True:
            match_episode_min -= 1
            if not os.path.exists(os.path.join(match_dataset_path, "obs", f"{match_episode_min}/0.npy")):
                break
        match_episode_min += 1
        match_episode_max = match_episode
        while True:
            match_episode_max += 1
            if not os.path.exists(os.path.join(match_dataset_path, "obs", f"{match_episode_max}/0.npy")):
                break
        match_episode_max -= 1
        print(f"match_episode: {match_episode}, match_episode_min: {match_episode_min}, match_episode_max: {match_episode_max}")


    ###################################################################################################################################################################################################
    if task_name == 'cup':
        no_mirror = True
    else:
        no_mirror = False
    ###################################################################################################################################################################################################

    tx_left_right = np.array(
        [
            [0.99996206, 0.00661996, 0.00566226, -0.01676012],
            [-0.00663261, 0.99997554, 0.0022186, -0.60552492],
            [-0.00564743, -0.00225607, 0.99998151, -0.007277],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    tx_robot1_robot0 = tx_left_right


    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith(".ckpt"):
        ckpt_path = os.path.join(ckpt_path, "checkpoints", "latest.ckpt")
    cfg_path = ckpt_path.replace(".ckpt", ".yaml")
    with open(cfg_path, "r") as f:
        cfg = OmegaConf.load(f)
    # import torch
    # payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    # cfg = payload['cfg']
    
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    print("obs_pose_rep", obs_pose_rep)
    print("action_pose_repr", action_pose_repr)

    max_obs_buffer_size = 1000

    if cfg.name=='uva':
        with open_dict(cfg):
            # cfg.task.shape_meta = cfg.task.dataset.shape_meta
            cfg.task.shape_meta.obs.camera0_rgb.horizon = 16
            cfg.task.shape_meta.obs.robot0_eef_pos.horizon = 16
        max_obs_buffer_size = 1000
    
    # setup experiment
    dt = 1 / frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)

    robots_config = [
        {
            "robot_type": "arx5",
            "robot_ip": "127.0.0.1",
            "robot_port": robot_port,
            "robot_obs_latency": 0.005,  # TODO: need to measure
            "robot_action_latency": 0.04,  # TODO: need to measure
            "height_threshold": -0.2,  # TODO: ncscseed to measure
            "sphere_radius": 0.1,  # TODO: need to measure
            "sphere_center": [0, -0.06, -0.185],  # TODO: need to measure
        }
    ]

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{policy_ip}:{policy_port}")

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(
            shm_manager=shm_manager, deadzone=0.1
        ) as sm, KeystrokeCounter() as key_counter, Arx5Env(
            output_dir=output,
            robots_config=robots_config,
            frequency=frequency,
            obs_image_resolution=obs_res,
            max_obs_buffer_size=max_obs_buffer_size,
            obs_float32=True,
            camera_reorder=[int(x) for x in camera_reorder],
            init_joints=init_joints,
            enable_multi_cam_vis=False,
            # latency
            camera_obs_latency=0.17,
            # obs
            camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
            robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
            no_mirror=no_mirror,
            # fisheye_converter=fisheye_converter,
            mirror_swap=mirror_swap,
            # action
            max_pos_speed=2.0,
            max_rot_speed=6.0,
            shm_manager=shm_manager,
        ) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            # time.sleep(3.0)

            print("Waiting for env ready.")
            while not env.is_ready:
                time.sleep(0.1)
            print("Env is ready")

            print(f"Warming up video recording")
            video_dir = env.video_dir.joinpath("test")
            video_dir.mkdir(exist_ok=True, parents=True)
            n_cameras = env.camera.n_cameras
            video_paths = []
            for i in range(n_cameras):
                video_path = str(video_dir.joinpath(f"{i}.mp4").absolute())
                video_paths.append(video_path)
            env.camera.start_recording(video_path=video_paths, start_time=time.time())
            time.sleep(3.0)

            print(f"Warming up policy inference")
            obs = env.get_obs()
            print(obs)

            episode_start_pose = list()
            for robot_id in range(len(robots_config)):
                pose = np.concatenate(
                    [
                        obs[f"robot{robot_id}_eef_pos"],
                        obs[f"robot{robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )[-1]
                episode_start_pose.append(pose)

            ####################################################
            ## reduce observation
            original_imgs = obs['camera0_rgb']
            if obs['camera0_rgb'].shape[0] > 4:
                T = 16
                indices = [3, 7, 11, 15]
                obs["camera0_rgb"] = obs['camera0_rgb'][indices, :, :, :]
                obs["robot0_eef_pos"] = obs["robot0_eef_pos"][indices]
                obs["robot0_eef_rot_axis_angle"] = obs["robot0_eef_rot_axis_angle"][indices]
                obs["robot0_gripper_width"] = obs["robot0_gripper_width"][indices]
                print('indices', indices)
            ####################################################


            obs_dict_np = get_real_umi_obs_dict(
                env_obs=obs,
                shape_meta=cfg.task.shape_meta,
                obs_pose_repr=obs_pose_rep,
                tx_robot1_robot0=tx_robot1_robot0,
                episode_start_pose=episode_start_pose,
            )
            
            
            obs_dict_np['task_name'] = task_name

            socket.send_pyobj(obs_dict_np)
            print(
                f"    obs_dict_np sent to PolicyInferenceNode at tcp://{policy_ip}:{policy_port}. Waiting for response."
            )
            start_time = time.monotonic()
            action = socket.recv_pyobj()
            if type(action) == str:
                print(
                    f"Inference from PolicyInferenceNode failed: {action}. Please check the model."
                )
                exit(1)
            print(
                f"Got response from PolicyInferenceNode. Inference time: {time.monotonic() - start_time:.3f} s"
            )

            env.camera.stop_recording()
            print(
                f"Warming up video recording finished. Video stored to {env.video_dir.joinpath(str(0))}"
            )

            assert action.shape[-1] == 10 * len(robots_config)
            action = get_real_umi_action(action, obs, action_pose_repr)
            assert action.shape[-1] == 7 * len(robots_config)

            print("Ready!")
            while True:
                last_control_is_human = True
                # ========= human control loop ==========
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs["ActualTCPPose"] for rs in robot_states])

                gripper_target_pos = np.asarray(
                    [rs["gripper_position"] for rs in robot_states]
                )

                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()


                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    os.makedirs(
                        os.path.join(output, "obs", f"{episode_id}"), exist_ok=True
                    )
                    os.makedirs(
                        os.path.join(output, "action", f"{episode_id}"), exist_ok=True
                    )
                    vis_img = obs[f"camera0_rgb"][-1]

                    if match_dataset_path is not None:
                        assert match_episode is not None
                        match_data = np.load(os.path.join(match_dataset_path, "obs", f"{match_episode}/0.npy"), allow_pickle=True).item()
                        obs_left_img = match_data["obs"]['camera0_rgb'][-1]
                        vis_img = np.concatenate(
                            [obs_left_img, vis_img], axis=1
                        )

                    text = f"Episode: {episode_id}, matching: {match_episode} ({match_dataset_path})"
                    cv2.putText(
                        vis_img,
                        text,
                        (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0, 0, 0),
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255, 255, 255),
                    )
                    cv2.imshow("default", vis_img[..., ::-1])
                    _ = cv2.pollKey()
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char="q"):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char="c"):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        elif key_stroke == KeyCode(char="e"):
                            # Next episode
                            if match_episode is not None:
                                match_episode = min(
                                    match_episode + 1, match_episode_max
                                )
                                print(f"match_episode: {match_episode}")
                        elif key_stroke == KeyCode(char="w"):
                            # Prev episode
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, match_episode_min)
                                print(f"match_episode: {match_episode}")
                        elif key_stroke == KeyCode(char="m"):
                            # Move robot to the starting pose in waiting_time seconds
                            waiting_time = 5.0
                            print("starting dataset smatching")

                            if match_episode is not None and match_dataset_path is not None:
                                # DEBUG: to be tested
                                match_data = np.load(os.path.join(match_dataset_path, "obs", f"{match_episode}/0.npy"), allow_pickle=True).item()
                                for robot_idx in range(len(robots_config)):
                                    start_pos = match_data["obs"][f"robot{robot_idx}_eef_pos"][-1]
                                    start_rot = match_data["obs"][f"robot{robot_idx}_eef_rot_axis_angle"][-1]
                                    target_pose = np.concatenate((start_pos, start_rot), axis=0)[np.newaxis, ...]

                                    gripper_target_pos = match_data["obs"][f"robot{robot_idx}_gripper_width"][-1]
                                action = np.zeros((7 * len(robots_config),))
                                print(target_pose)
                                for robot_idx in range(len(robots_config)):
                                    action[7 * robot_idx + 0 : 7 * robot_idx + 6] = target_pose[robot_idx]
                                    action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]
                                env.exec_actions(
                                    actions=[action],
                                    timestamps=[t_command_target - time.monotonic() + time.time() + waiting_time],
                                    compensate_latency=False,
                                )
                                print(f"executing action: {action}")
                                precise_wait(t_cycle_end + waiting_time)
                                iter_idx += int(waiting_time / dt)
                                continue
                            
                        elif key_stroke == Key.backspace:
                            if click.confirm("Are you sure to drop an episode?"):
                                env.drop_episode()
                                key_counter.clear()
                        elif key_stroke == KeyCode(char="a"):
                            control_robot_idx_list = list(range(target_pose.shape[0]))
                        elif key_stroke == KeyCode(char="1"):
                            control_robot_idx_list = [0]
                        elif key_stroke == KeyCode(char="2"):
                            control_robot_idx_list = [1]
                    
                    if start_policy:
                        break
                    precise_wait(t_sample)
                    # get teleop command
                    sm_state = sm.get_motion_state_transformed()
                    # sm_state = get_filtered_spacemouse_output(sm)
                    # print(sm_state)
                    dpos = sm_state[:3] * (0.5 / frequency) * cartesian_speed
                    drot_xyz = sm_state[3:] * (1.5 / frequency) * orientation_speed

                    drot = st.Rotation.from_euler("xyz", drot_xyz)
                    for robot_idx in control_robot_idx_list:
                        target_pose[robot_idx, :3] += dpos
                        target_pose[robot_idx, 3:] = (
                            drot * st.Rotation.from_rotvec(target_pose[robot_idx, 3:])
                        ).as_rotvec()
                        # target_pose[robot_idx, 2] = np.maximum(target_pose[robot_idx, 2], 0.055)

                    dpos = 0
                    if sm.is_button_pressed(0) and sm.is_button_pressed(1):
                        print("Reset robot arm to home. Please wait...")
                        for robot_idx in control_robot_idx_list:
                            env.robots[robot_idx].reset_to_home()
                            robot_states = env.get_robot_state()
                            target_pose[robot_idx] = np.stack(
                                [rs["ActualTCPPose"] for rs in robot_states]
                            )
                            gripper_target_pos[robot_idx] = np.asarray(
                                [rs["gripper_position"] for rs in robot_states]
                            )

                    elif sm.is_button_pressed(0):
                        # close gripper
                        dpos = -gripper_speed / frequency
                    elif sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency
                    for robot_idx in control_robot_idx_list:
                        gripper_target_pos[robot_idx] = np.clip(
                            gripper_target_pos[robot_idx] + dpos, 0, max_gripper_width
                        )

                    # # solve collision with table
                    # for robot_idx in control_robot_idx_list:
                    #     solve_table_collision(
                    #         ee_pose=target_pose[robot_idx],
                    #         gripper_width=gripper_target_pos[robot_idx],
                    #         height_threshold=robots_config[robot_idx]['height_threshold'])

                    # # solve collison between two robots
                    # solve_sphere_collision(
                    #     ee_poses=target_pose,
                    #     robots_config=robots_config
                    # )

                    action = np.zeros((7 * target_pose.shape[0],))

                    for robot_idx in range(target_pose.shape[0]):
                        action[7 * robot_idx + 0 : 7 * robot_idx + 6] = target_pose[
                            robot_idx
                        ]
                        action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]

                    # execute teleop command
                    env.exec_actions(
                        actions=[action],
                        timestamps=[t_command_target - time.monotonic() + time.time()],
                        compensate_latency=False,
                    )
                    precise_wait(t_cycle_end)
                    iter_idx += 1

                # ========== policy control loop ==============
                try:
                    # start episode
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # get current pose
                    obs = env.get_obs()

                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate(
                            [
                                obs[f"robot{robot_id}_eef_pos"],
                                obs[f"robot{robot_id}_eef_rot_axis_angle"],
                            ],
                            axis=-1,
                        )[-1]
                        episode_start_pose.append(pose)



                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1 / 60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        # if no_mirror:
                        #     for i, camera0_rgb in enumerate(obs[f"camera0_rgb"]):
                        #         obs[f"camera0_rgb"][i] = draw_predefined_mask(
                        #             camera0_rgb, (0,0,0), mirror=no_mirror, gripper=False, finger=False, use_aa=True
                        #         )
                        obs_timestamps = obs["timestamp"]
                        print(f"Obs latency {time.time() - obs_timestamps[-1]}")
                        if np.mean(obs["camera0_rgb"][-1]) < 0.1:
                            raise RuntimeError("Camera not connected")
                        # # HACK: Only use the last image    
                        # obs["camera0_rgb"] = np.array([obs["camera0_rgb"][-1] for _ in obs["camera0_rgb"]])
                        if last_control_is_human:
                            # Pad the entire observatioscsn with the last frame
                            print(f"padding using the last observation")
                            for k, v in obs.items():
                                if k != 'timestamp':
                                    original_length = v.shape[0]
                                    obs[k] = np.concatenate([v[-1:]] * original_length, axis=0)

                        last_control_is_human = False
                        ####################################################
                        ## reduce observation
                        original_imgs = obs['camera0_rgb']
                        if obs['camera0_rgb'].shape[0] > 4:
                            T = 16
                            # select_timesteps = 4
                            # indices = np.arange(0, T, step=T//select_timesteps) + select_timesteps - 1
                            # indices = [12, 13, 14, 15]
                            indices = [3, 7, 11, 15]
                            obs["camera0_rgb"] = obs['camera0_rgb'][indices, :, :, :]
                            
                            obs["robot0_eef_pos"] = obs["robot0_eef_pos"][indices]
                            obs["robot0_eef_rot_axis_angle"] = obs["robot0_eef_rot_axis_angle"][indices]
                            obs["robot0_gripper_width"] = obs["robot0_gripper_width"][indices]
                            print('indices', indices)
                        ####################################################

                        # run inference
                        s = time.time()
                        obs_dict_np = get_real_umi_obs_dict(
                            env_obs=obs,
                            shape_meta=cfg.task.shape_meta,
                            obs_pose_repr=obs_pose_rep,
                            tx_robot1_robot0=tx_robot1_robot0,
                            episode_start_pose=episode_start_pose,
                        )
                        
                        obs_dict_np['task_name'] = task_name


                        obs_data = {
                            "obs_dict_np": obs_dict_np,
                            "obs_pose_rep": obs_pose_rep,
                            "obs": obs,
                            "original_imgs": original_imgs,
                            "episode_start_pose": episode_start_pose,
                            "tx_robot1_robot0": tx_robot1_robot0,
                        }
                        np.save(
                            os.path.join(
                                output, "obs", f"{episode_id}", f"{iter_idx}.npy"
                            ),
                            obs_data,
                            allow_pickle=True,
                        )
                        
                        
                        socket.send_pyobj(obs_dict_np)
                        raw_action = socket.recv_pyobj()

                        # print(raw_action[:, :3])

                        for k, raw_action_cmd in enumerate(raw_action):
                            translation = raw_action_cmd[:3]
                            rotation = st.Rotation.from_matrix(rot6d_to_mat(raw_action_cmd[3:9])).as_rotvec()
                            if np.max(np.abs(translation)) / (k+1) > 0.1:
                                print(f"==============={np.mean(obs['camera0_rgb'][-1])}================")
                                print(f"  {translation}, {rotation}")

                                raise RuntimeError("Action translation too large. Please check the input")

                        
                        if type(raw_action) == str:
                            print(
                                f"Inference from PolicyInferenceNode failed: {raw_action}. Please check the model."
                            )
                            env.end_episode()
                            break
                        action = get_real_umi_action(raw_action, obs, action_pose_repr)
                        action_data = {
                            "action": action,
                            "raw_action": raw_action,
                            "action_pose_repr": action_pose_repr,
                        }
                        np.save(
                            os.path.join(
                                output, "action", f"{episode_id}", f"{iter_idx}.npy"
                            ),
                            action_data,
                            allow_pickle=True,
                        )
                        print("Inference latency:", time.time() - s)

                        # convert policy action to env actions
                        this_target_poses = action
                        assert this_target_poses.shape[1] == len(robots_config) * 7
                        # for target_pose in this_target_poses:
                        #     for robot_idx in range(len(robots_config)):
                        #         solve_table_collision(
                        #             ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                        #             gripper_width=target_pose[robot_idx * 7 + 6],
                        #             height_threshold=robots_config[robot_idx]['height_threshold']
                        #         )

                        #     # solve collison between two robots
                        #     solve_sphere_collision(
                        #         ee_poses=target_pose.reshape([len(robots_config), -1]),
                        #         robots_config=robots_config
                        #     )

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (
                            np.arange(len(action), dtype=np.float64)
                        ) * dt + obs_timestamps[-1]
                        # action_exec_latency = 0.01
                        # curr_time = time.time()
                        # is_new = action_timestamps > (curr_time + action_exec_latency)
                        # if np.sum(is_new) == 0:
                        #     # exceeded time budget, still do something
                        #     this_target_poses = this_target_poses[[-1]]
                        #     # schedule on next available step
                        #     next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                        #     action_timestamp = eval_t_start + (next_step_idx) * dt
                        #     print('Over budget', action_timestamp - curr_time)
                        #     action_timestamps = np.array([action_timestamp])
                        # else:
                        #     this_target_poses = this_target_poses[is_new]
                        #     action_timestamps = action_timestamps[is_new]

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            # compensate_latency=True
                            dynamic_latency=True,
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        obs_left_img = obs["camera0_rgb"][-1]
                        obs_right_img = obs["camera0_rgb"][-1]
                        vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
                        text = "Episode: {}, Time: {:.1f}".format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255, 255, 255),
                        )
                        cv2.imshow("default", vis_img[..., ::-1])

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char="s"):
                                # Stop episode
                                # Hand control back to human
                                print("Stopped.")
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()

                print("Stopped.")


# %%
if __name__ == "__main__":
    main()
