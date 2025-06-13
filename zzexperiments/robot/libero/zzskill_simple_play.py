"""Replay the trajectory stored in HDF5.
The replayed trajectory can use different observation modes and control modes.
We support translating actions from certain controllers to a limited number of controllers.
The script is only tested for Panda, and may include some Panda-specific hardcode.
"""

import argparse
import multiprocessing as mp
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


def main(maniskill_cfg, proc_id: int = 0, num_procs=1, pbar=None):
    # Load HDF5 containing trajectories
    path = "/home/hz/code/PointCloudMatters/data/maniskill2/demos/v0/rigid_body/StackCube-v0/trajectory.rgbd.pd_ee_delta_pose.h5"
    group_name = f"traj_{proc_id}"  
    data = read_hdf5_group(path, group_name)
    traj_path = maniskill_cfg.traj_path
    ori_h5_file = h5py.File(traj_path, "r")


    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]

    target_obs_mode = maniskill_cfg.obs_mode
    target_control_mode = maniskill_cfg.target_control_mode
    env_kwargs = ori_env_kwargs.copy()
    if target_obs_mode is not None:
        env_kwargs["obs_mode"] = target_obs_mode
    if target_control_mode is not None:
        env_kwargs["control_mode"] = target_control_mode
    env_kwargs["bg_name"] = maniskill_cfg.bg_name
    env_kwargs["render_mode"] = "rgb_array"  
    env = gym.make(env_id, **env_kwargs)

    env.reset()

    if maniskill_cfg.vis:
        env.render_human()
    
    for action in tqdm(data['actions'], desc="Replaying actions"):
        obs, _, _, _, info = env.step(action)
        if maniskill_cfg.vis:
            env.render_human()


class ManiskillCfg:
    obs_mode = "pointcloud"
    traj_path = "/home/hz/code/PointCloudMatters/data/maniskill2/demos/v0/rigid_body/StackCube-v0/trajectory.h5"
    target_control_mode = "pd_ee_delta_pose"
    bg_name = None
    vis = False


if __name__ == "__main__":
    maniskill_cfg = ManiskillCfg()
    main(maniskill_cfg)
