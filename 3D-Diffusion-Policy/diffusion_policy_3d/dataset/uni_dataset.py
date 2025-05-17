from typing import Dict, List
import os
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from pathlib import Path

class UniDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
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
        self.zarr_path = zarr_path
        # zarr_paths = self.zarr_path
        
        zarr_paths = self.get_subdirs_with_path(self.zarr_path)
        # import pdb; pdb.set_trace()

        # self.replay_buffers = []
        # for path in zarr_paths:
        #     buffer = ReplayBuffer.copy_from_path(
        #         path, keys=['state', 'action', 'point_cloud', 'img'])
        #     self.replay_buffers.append(buffer)
        # import pdb; pdb.set_trace()
        self.replay_buffer = self._merge_replay_buffers(zarr_paths)
        # import pdb; pdb.set_trace()
        # self.replay_buffer = ReplayBuffer.copy_from_path(
        #     zarr_paths, keys=['state', 'action', 'point_cloud', 'img']) # 数据读取成功


        # 3. 划分训练/验证集（跨数据集统一划分）
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        # val_mask = get_val_mask(
        #     n_episodes=self.replay_buffer.n_episodes,  # 这里就是采集的demo数量
        #     val_ratio=val_ratio,
        #     seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes,  # 这里是最大训练的episode数量，不过貌似一般不会超过这个数值，这个函数保证训练的episode不超过预设
            seed=seed)
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)


        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_subdirs_with_path(self, parent_dir):
        path = Path(parent_dir)
        all_paths = [str(child) for child in path.iterdir() if child.is_dir()]

        # 分配给每个进程一个子路径（可选按 rank）
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        if world_size > 1:
            all_paths.sort()
            sub_paths = all_paths[rank::world_size]  # 每个 rank 处理部分路径
            return sub_paths
        else:
            return all_paths



    def _merge_replay_buffers(self, zarr_paths: List[str]) -> ReplayBuffer:
        """
        合并多个 zarr 文件到一个 ReplayBuffer。
        """
        print(f"merging {zarr_paths} zarr files")
        buffers = []
        for path in zarr_paths:
            buffer = ReplayBuffer.copy_from_path(
                path, keys=['state', 'action', 'point_cloud'])
            buffers.append(buffer)
        # import pdb; pdb.set_trace()
        if not buffers:
            raise ValueError("No valid zarr paths provided")
        # buffers[0]['point_cloud'].shape
        merged_buffer = ReplayBuffer.create_empty_numpy()  # 或 create_empty_zarr()

        # 3. 合并数据（逐个 episode 添加）
        for buf in buffers:
            for episode_idx in range(buf.n_episodes):
                # 获取单个 episode 的数据
                episode_data = buf.get_episode(episode_idx)
                # 添加到 merged_buffer
                merged_buffer.add_episode(episode_data)
        # import pdb; pdb.set_trace()
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

    def get_normalizer(self, mode='limits', device='cpu', **kwargs):
        # 从 replay buffer 构造数据
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][..., :],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        device = torch.device(f"cuda:{device}")
        # 将所有张量或数组转为 torch.Tensor 并搬到目标 device
        def to_device(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            return x.to(device)

        data = {k: to_device(v) for k, v in data.items()}

        # 拟合 normalizer
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        return normalizer


    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)  # 这里只有两个key obs(point_cloud, agent_pos)和action (16,7)
        # (16,1024,6)  (16,7)这个agent_pos是啥
        # import pdb; pdb.set_trace()
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

