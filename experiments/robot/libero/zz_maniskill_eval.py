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

import gymnasium as gym
import h5py
import numpy as np
import sapien.core as sapien
from tqdm.auto import tqdm
from transforms3d.quaternions import quat2axangle

import mani_skill2.envs
from mani_skill2.agents.base_controller import CombinedController
from mani_skill2.agents.controllers import *
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.trajectory.merge_trajectory import merge_h5
from mani_skill2.utils.common import clip_and_scale_action, inv_scale_action
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import get_entity_by_name
from mani_skill2.utils.wrappers import RecordEpisode

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


def get_maniskill_observation(last_obs, obs, ee_pose_traj, step):
    dummy_action = get_maniskill_dummy_action()
    
    last_pcd = np.array(last_obs["pointcloud"]["xyzw"][..., :3], dtype=float)
    last_pcd = downsample_with_fps(last_pcd)
    
    pcd = np.array(obs["pointcloud"]["xyzw"][..., :3], dtype=float)
    pcd = downsample_with_fps(pcd)
    
    last_state = np.array(last_obs['agent']['qpos'], dtype=float)
    state = np.array(obs['agent']['qpos'], dtype=float)
    
    if step + 16 > len(ee_pose_traj):
        ee_pos = ee_pose_traj[-1]
        ee_pos= np.array([ee_pos] * 16)
    else:
        ee_pos = ee_pose_traj[step:step+16, ...]
    res_ee_pos = torch.tensor([ee_pos])
    
    res_pcd = torch.tensor(np.array([last_pcd, pcd])).unsqueeze(0)
    res_state = torch.tensor(np.array([last_state, state])).unsqueeze(0)
    
    noise_action = torch.tensor(np.array([dummy_action] * 16)).unsqueeze(0)

    observation = {
        "obs": {
            'point_cloud': res_pcd,
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
    
    


def maniskill_run(cfg: GenerateConfig, maniskill_cfg, model, proc_id: int = 0):
    # Load HDF5 containing trajectories
    path = "/home/hz/code/PointCloudMatters/data/maniskill2/demos/v0/rigid_body/StackCube-v0/trajectory.rgbd.pd_ee_delta_pose.h5"
    group_name = f"traj_{proc_id}"  
    data = read_hdf5_group(path, group_name)
    traj_path = maniskill_cfg.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    eepose_traj = get_maniskill_eepose_traj(data)
    
    data_actions = data['actions']

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]

    target_obs_mode = maniskill_cfg.obs_mode
    target_control_mode = maniskill_cfg.target_control_mode
    env_kwargs = ori_env_kwargs.copy()
    env_kwargs["obs_mode"] = target_obs_mode
    env_kwargs["control_mode"] = target_control_mode
    env_kwargs["bg_name"] = maniskill_cfg.bg_name
    env_kwargs["render_mode"] = "rgb_array"  
    env = gym.make(env_id, **env_kwargs)
    # env = gym.Wrapper(env)
    env = RecordEpisode(
        env,
        "./zzskill_simple_eval",
        save_on_reset=True,
        save_trajectory=False,
        save_video=True
    )

    env.reset(seed=maniskill_cfg.env_reset_seed)

    if maniskill_cfg.vis:
        env.render_human()
    
    t = 0
    max_steps = 320
    dummy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
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
        
        action = get_dp3_action(model, observation)
        action = action['action'].detach().numpy()

        for (i, step_action) in enumerate(action[0]):
            last_obs = obs
            obs, _, _, _, info = env.step(step_action)
            # obs, _, _, _, info = env.step(data_actions[t - cfg.num_steps_wait % len(data_actions)])
            t += 1
        success = info.get("success", False)
        if success:
            break
        print(info)
        
        if maniskill_cfg.vis:
            env.render_human()
    
    env.flush_video()
        
        


class ManiskillCfg:
    obs_mode = "pointcloud"
    traj_path = "/home/hz/code/PointCloudMatters/data/maniskill2/demos/v0/rigid_body/StackCube-v0/trajectory.h5"
    target_control_mode = "pd_ee_delta_pose"
    bg_name = None
    vis = False
    env_reset_seed = 0

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
    
    maniskill_cfg = ManiskillCfg()
    maniskill_run(cfg, maniskill_cfg, model)


if __name__ == "__main__":
    eval_maniskill()
