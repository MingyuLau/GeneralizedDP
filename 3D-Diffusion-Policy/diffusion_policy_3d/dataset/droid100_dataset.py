from typing import Dict, List
import torch
import numpy as np
import copy
import tensorflow_datasets as tfds
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from pathlib import Path

class DroidDataset(BaseDataset):
    def __init__(self,
            data_dir,  # DROID数据集的路径
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
       
        self.task_name = task_name
        self.data_dir = data_dir
        
        # 加载DROID数据集并转换为ReplayBuffer格式
        self.replay_buffer = self._load_droid_data()
        
        # 划分训练/验证集
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes,
            seed=seed)
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        # import pdb; pdb.set_trace()
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _load_droid_data(self) -> ReplayBuffer:
        """
        从DROID数据集加载数据并转换为ReplayBuffer格式
        """
        # 加载DROID数据集
        ds = tfds.load("droid_100", data_dir=self.data_dir, split="train")
        
        merged_buffer = ReplayBuffer.create_empty_numpy()
        
        for episode in ds:
            episode_data = {
                'action': [],           # 对应 cartesian_velocity
                'agent_pos': [],        # 对应 joint_position  
                'ee_pos': []           # 对应 cartesian_position + (1 - gripper_position)
            }
            
            # 提取每个episode中的步骤数据
            for step in episode["steps"]:
                # 提取 action_dict
                action_dict = step['action_dict']
                
                # action 对应 cartesian_velocity (3维)
                cartesian_vel = action_dict['cartesian_velocity'].numpy().astype(np.float32)
                gripper_pos = action_dict['gripper_position'].numpy().astype(np.float32)      # 1维
                gripper_openness = 1.0 - gripper_pos  # 计算夹爪开合度
                gripper_openness = (gripper_openness > 0.5).astype(np.float32)
                cartesian_vel = np.concatenate([cartesian_vel, gripper_openness], axis=-1)
                episode_data['action'].append(cartesian_vel)
                
                # agent_pos 对应 joint_position (7维)
                joint_pos = action_dict['joint_position'].numpy().astype(np.float32)
                episode_data['agent_pos'].append(joint_pos)
                
                # ee_pos 对应 cartesian_position + (1 - gripper_position)
                cartesian_pos = action_dict['cartesian_position'].numpy().astype(np.float32)  # 3维
                
                
                # 组合成 ee_pos (4维: 3维位置 + 1维夹爪)
                ee_pos = np.concatenate([cartesian_pos, gripper_openness], axis=-1)
                episode_data['ee_pos'].append(ee_pos)
            
            # 转换为numpy数组
            for key in episode_data:
                episode_data[key] = np.array(episode_data[key])
            # 创建episode数据字典
            
            episode_dict = {
                'action':  episode_data['ee_pos'],      # (T, 3) - cartesian_velocity
                'state': episode_data['agent_pos'],    # (T, 7) - joint_position
                'ee_pos': episode_data['ee_pos']       # (T, 4) - cartesian_position + gripper_openness
            }
            
            # 添加到buffer
            merged_buffer.add_episode(episode_dict)
    
        return merged_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # import  pdb; pdb.set_trace()
        data = {
            'action': self.replay_buffer['ee_pos'],      # cartesian_velocity (3维)
            'agent_pos': self.replay_buffer['state'],    # joint_position (7维)
            'ee_pos': self.replay_buffer['ee_pos'],      # cartesian_position + gripper (4维)
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_data(self, mode='limits', **kwargs):
        # import  pdb; pdb.set_trace()
        data = {
            'action': self.replay_buffer['ee_pos'],      # cartesian_velocity
            'agent_pos': self.replay_buffer['state'],    # joint_position  
            'ee_pos': self.replay_buffer['ee_pos'],      # cartesian_position + gripper
        }
        return data

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32)     # joint_position (T, 7)
        ee_pos = sample['ee_pos'].astype(np.float32)       # cartesian_pos + gripper (T, 4)
        # import pdb; pdb.set_trace()
        data = {
            'obs': {
                'agent_pos': agent_pos,    # T, 7 (joint_position)
                # 'ee_pos': ee_pos,          # T, 4 (cartesian_position + gripper_openness)
            },
            'action': ee_pos  # T, 3 (cartesian_velocity)
        }
        return data
    
    def __len__(self) -> int:
        return len(self.sampler)

    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data