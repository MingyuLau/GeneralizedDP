"""Replay the trajectory stored in HDF5.
The replayed trajectory can use different observation modes and control modes.
We support translating actions from certain controllers to a limited number of controllers.
The script is only tested for Panda, and may include some Panda-specific hardcode.
"""

import argparse
import multiprocessing as mp
from multiprocessing import dummy
import os
from copy import deepcopy
from typing import Union

from flask import wrappers
import gymnasium as gym
import h5py
import numpy as np
from requests import get
import sapien.core as sapien
from tqdm.auto import tqdm
from transforms3d.quaternions import quat2axangle

from mani_skill.utils.io_utils import load_json
from mani_skill.utils.wrappers import RecordEpisode

# from mani_skill.agents.controllers import get_controller

import h5py

import os
from re import I
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from collections import deque
sys.path.append("/home/hz/code/GeneralizedDP/3D-Diffusion-Policy")
sys.path.append("/home/hz/code/GeneralizedDP/experiments/")
sys.path.append("/home/hz/code/GeneralizedDP/")
# from experiments.debug_utils import setup_debug
# import wandb
import cv2
# Append current directory so that interpreter can find experiments.robot
# sys.path.append("../..")
from zzexperiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
# from experiments.robot.openvla_utils import get_processor
from zzexperiments.robot_utils import (
    DATE_TIME,
    get_latent_action,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
    get_dp3_action,
)

import numpy as np
import pytorch3d.ops as torch3d_ops

from plyfile import PlyData, PlyElement

from mani_skill.utils import wrappers


def downsample_with_fps(points: np.ndarray, num_points: int = 1024):
    # fast point cloud sampling using torch3d
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points = torch.tensor([num_points]).cuda()
    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=num_points)
    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]
    return points

def read_hdf5_group(file_path, group_name):
    def read_group(group):
        group_data = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                # 如果是子组，递归读取
                group_data[key] = read_group(item)
            elif isinstance(item, h5py.Dataset):
                # 如果是数据集，读取数据
                group_data[key] = item[:]
        return group_data

    with h5py.File(file_path, 'r') as f:
        if group_name in f:
            group = f[group_name]
            return read_group(group)
        else:
            print(f"Group '{group_name}' not found in the file.")
            return None





@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "dp3"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "./vla-scripts/libero_log/finetune-libero"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    
    action_decoder_path:str = "./vla-scripts/libero_log/finetune-libero/action_decoder.pt"
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    save_video: bool = False                         # Whether to save rollout videos

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "maniskill_stackcube"               # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    window_size: int = 12

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/eval_logs"   # Local directory for eval logs
    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    image_history_size: int = 2
    

def get_maniskill_dummy_action():
    dummy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return dummy_action


def save_point_cloud_to_ply(point_cloud, file_path):
    # 确保颜色值在0-255范围内
    colors = point_cloud[:, 3:6].astype(np.uint8)
    # 提取坐标
    vertices = point_cloud[:, 0:3]
    
    # 创建结构化数组
    vertex_data = np.zeros(len(vertices), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]
    
    # 创建PLY元素
    el = PlyElement.describe(vertex_data, 'vertex')
    # 创建并保存PLY文件
    PlyData([el], text=False).write(file_path)
    
    print(f"点云已保存到 {file_path}")


def get_maniskill_observation(last_obs, obs, ee_pose_traj, step):
    dummy_action = get_maniskill_dummy_action()
    
    # last_pcd = np.array(last_obs["pointcloud"]["xyzw"][..., :3].cpu(), dtype=float)
    # last_pcd = downsample_with_fps(last_pcd)
    
    # pcd = np.array(obs["pointcloud"]["xyzw"][..., :3], dtype=float)
    # pcd = downsample_with_fps(pcd)
    
    # pcd_rgb = np.array(obs["pointcloud"]["rgb"])
    # pcd_rgb = downsample_with_fps(pcd_rgb)
    
    # point_cloud_save = np.concatenate([pcd, pcd_rgb], axis=-1)
    
    # save_point_cloud_to_ply(point_cloud_save, f"./point_cloud_{step}.ply")
    
    last_state = np.array(last_obs['agent']['qpos'], dtype=float).squeeze()
    state = np.array(obs['agent']['qpos'], dtype=float).squeeze()
    # import pdb; pdb.set_trace()
    if step + 16 > len(ee_pose_traj):
        ee_pos = ee_pose_traj[-1]
        ee_pos= np.array([ee_pos] * 16)
        # ee_pos = np.array([[0,0,0,0,0,0,0]] * 16, dtype=float)
        # ee_pos = np.array([[1,1,1,1,1,1,1]] * 16, dtype=float)
    else:
        ee_pos = ee_pose_traj[step:step+16, ...]
        # ee_pos = np.array([[0,0,0,0,0,0,0]] * 16, dtype=float)
        # ee_pos = np.array([[1,1,1,1,1,1,1]] * 16, dtype=float)
    res_ee_pos = torch.tensor(np.array([ee_pos]))
    
    # res_pcd = torch.tensor(np.array([last_pcd, pcd])).unsqueeze(0)
    res_state = torch.tensor(np.array([last_state, state])).unsqueeze(0)
    
    noise_action = torch.tensor(np.array([dummy_action] * 16)).unsqueeze(0)

    observation = {
        "obs": {
            # 'point_cloud': res_pcd,
            'agent_pos': res_state,
            "ee_pos": res_ee_pos,
        },
        "actions": noise_action
    }
    
    return observation



def convert_action_state(tcp_poses, actions):
    def quaternion_to_euler(quaternion):
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(quaternion)
        return r.as_euler('ZYX', degrees=False)
    tcp_states = np.array([quaternion_to_euler(tcp_pose[-4:]) for tcp_pose in tcp_poses])
    tcp_states = np.concatenate([tcp_states, actions[:, -1, None]], axis=-1)
    tcp_states = np.concatenate([tcp_poses[:, :3], tcp_states], axis=-1)
    
    return tcp_states


def get_maniskill_eepose_traj(data):
    eepose_traj = None
    tcp_poses = data["obs"]["extra"]["tcp_pose"]
    actions = data["actions"]
    if len(actions) < len(tcp_poses):
        actions = np.concatenate([actions, [get_maniskill_dummy_action()]], axis=0)

    eepose_traj = convert_action_state(tcp_poses, actions)
    return eepose_traj


def get_maniskill_zarr_eepose_traj(action, tcp_poses):
    eepose_traj = None

    if len(action) < len(tcp_poses):
        action = np.concatenate([action, [action[-1]]], axis=0)

    eepose_traj = convert_action_state(tcp_poses, action)
    return eepose_traj



class ManiskillCfg:
    obs_mode = "pointcloud"
    traj_json_path = "/home/hz/data/dataset/maniskill/ManiSkill_Demonstrations/demos/StackCube-v1/motionplanning/trajectory.json"
    # traj_path = "/home/hz/code/PointCloudMatters/data/maniskill2/demos/v0/rigid_body/StackCube-v0/trajectory.rgbd.pd_ee_delta_pose.h5"
    traj_path = "/home/hz/code/GeneralizedDP/3D-Diffusion-Policy/data/dataset/maniskill/StackCube"
    # target_control_mode = "pd_ee_delta_pose"
    target_control_mode = "pd_ee_pose"
    # bg_name = None
    vis = False
    # vis = True
    env_reset_seed = 0
    env_kwargs = {
        'obs_mode': 'pointcloud', 
        'control_mode': 'pd_ee_pose', 
        'render_mode': 'rgb_array', 
        'reward_mode': None, 
        'sensor_configs': {'shader_pack': 'default'}, 
        'human_render_camera_configs': {'shader_pack': 'default'}, 
        'viewer_camera_configs': {'shader_pack': 'default'}, 
        'sim_backend': 'physx_cpu', 
        'num_envs': 1
    }
    output_dir = "./zzskill3_simple_eval"
    record_episode_kwargs = {
        'save_on_reset': True, 
        'save_trajectory': False, 
        'save_video': True, 
        'record_reward': False
    }
    env_id = "StackCube-v1"

    def __init__(self, proc_id=0):
        self.proc_id = proc_id
        
        data = load_json(self.traj_json_path)

        episodes = data["episodes"]
        self.env_reset_seed = episodes[proc_id]["episode_seed"]
            
        


def maniskill_run(cfg: GenerateConfig, maniskill_cfg: ManiskillCfg, model, proc_id: int = 0):
    # Load HDF5 containing trajectories
    
    path = maniskill_cfg.traj_path
    group_name = f"traj_{proc_id}"  
    
    # load data from zarr file
    data = None
    import zarr
    data = zarr.open(path, mode='r')
    actions_buffer = list(data['data']['action'])
    tcp_pose_buffer = list(data['data']['state'])
    episode_ends = list(data['meta']['episode_ends'])
    episode_ends = [0] + episode_ends
    episode_start = episode_ends[proc_id]
    episode_ends = episode_ends[proc_id+1]
    data_actions = np.array(actions_buffer[episode_start:episode_ends])
    tcp_poses = np.array(tcp_pose_buffer[episode_start:episode_ends])

    eepose_traj = get_maniskill_zarr_eepose_traj(data_actions, tcp_poses)

    env_id = maniskill_cfg.env_id

    env = gym.make(env_id, **maniskill_cfg.env_kwargs)
    assert env is not None
    # env = gym.Wrapper(env)
    env = wrappers.RecordEpisode(
        env = env,
        output_dir=maniskill_cfg.output_dir,
        trajectory_name= f"{env_id}_{proc_id}",
        video_fps=(
            env.unwrapped.control_freq
        ),
        **maniskill_cfg.record_episode_kwargs
    )

    env.reset(seed=maniskill_cfg.env_reset_seed)

    if maniskill_cfg.vis:
        env.render_human()
    
    t = 0
    max_steps = 200
    dummy_action = data_actions[0]
    obs = None
    last_obs = None
    
    while t < max_steps + cfg.num_steps_wait:
        
        if t < cfg.num_steps_wait:
            last_obs = obs
            
            obs, _, _, _, info = env.step(dummy_action)
            t += 1
            continue
        
        assert obs is not None
        
        observation = get_maniskill_observation(last_obs, obs, eepose_traj, t-cfg.num_steps_wait)
        gt_state_1 = tcp_poses[t - cfg.num_steps_wait if t - cfg.num_steps_wait < len(tcp_poses) else -1][:9]
        gt_state_2 = tcp_poses[t - cfg.num_steps_wait + 1 if t - cfg.num_steps_wait + 1 < len(tcp_poses) else -1][:9]
        gt_state = np.array([gt_state_1, gt_state_2])
        observation['obs']['agent_pos'] = torch.tensor(np.array([gt_state]), dtype=torch.float64)


        action = get_dp3_action(model, observation)
        action = action['action'].detach().numpy()
        # import pdb; pdb.set_trace()
        try:
            for (i, step_action) in enumerate(action[0]):
                last_obs = obs
                # step_action = np.clip(step_action, -3.139, 3.139)
                # import pdb; pdb.set_trace()
                # obs, _, _, _, info = env.step(step_action)
                # print("DP3 action:", step_action)
                # print("gt action:", data_actions[t - cfg.num_steps_wait % len(data_actions)])
                # obs, _, _, _, info = env.step(data_actions[t - cfg.num_steps_wait % len(data_actions)])
                obs, _, _, _, info = env.step(step_action)
                t += 1
        except:
            break
        success = info.get("success", False)
        if success:
            break
        # print(info)
        
        if maniskill_cfg.vis:
            env.render_human()
            
    return success
    
        
        

from tqdm import tqdm

@draccus.wrap()
def eval_maniskill(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)
    
    cfg.unnorm_key = cfg.task_suite_name
    
    model = get_model(cfg)
    
    pbar = tqdm(range(999), desc="Evaluating episodes")
    
    total_success = 0
    failed_list = []
    
    try:
        for i in range(999):
            maniskill_cfg = ManiskillCfg(i)
            proc_id = i
            success = maniskill_run(cfg, maniskill_cfg, model, proc_id)
            if success:
                total_success += 1
            else:
                failed_list.append(proc_id)
            # 计算成功率
            success_rate = (total_success / (i + 1)) * 100
            
            # 更新进度条，显示成功数量和成功率
            pbar.set_postfix({
                'Success': total_success,
                'Rate': f'{success_rate:.1f}%'
            })
            pbar.update(1)
    except:
        print("An error occurred during evaluation. Continuing with remaining tasks.")

    # 把失败的任务ID写入文件
    failed_file_path = os.path.join(cfg.local_log_dir, "failed_tasks_gt.txt")
    with open(failed_file_path, 'w') as f:
        for task_id in failed_list:
            f.write(f"{task_id}\n")
    pbar.close()


if __name__ == "__main__":
    eval_maniskill()
