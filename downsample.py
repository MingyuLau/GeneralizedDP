import os
import zarr
import numpy as np
import pytorch3d.ops as torch3d_ops
import torch

def downsample_with_fps(points: np.ndarray, num_points: int = 1024):
    """
    使用 farthest point sampling 对点云进行降采样
    :param points: 输入的点云数据 (numpy 数组)
    :param num_points: 降采样后的点数
    :return: 降采样后的点云和采样点的索引
    """
    points = torch.from_numpy(points).cuda()
    flatten_points = []
    
    for i in range(points.shape[0]):
        total_points = points.shape[1] * points.shape[2]
        if num_points > total_points:
            raise ValueError(f"num_points ({num_points}) cannot be greater than total points in the cloud ({total_points})")

        flatten_point = points[i, :, :, :3].reshape(1, 128 * 128, 3)  # Flatten to (1, P, 3)

        _, sampled_indices = torch3d_ops.sample_farthest_points(points=flatten_point, K=torch.tensor([num_points]).cuda())
        flatten_point = flatten_point[0, sampled_indices.squeeze(0).cpu().numpy()]
        flatten_points.append(flatten_point.cpu().numpy())

    return flatten_points, sampled_indices.squeeze(0).cpu().numpy()  # 返回降采样后的点云和索引

def recursive_read_zarr(group):
    """
    递归读取 Zarr 数据，并打印每个数据集的名称和数据。
    
    :param group: 当前的 Zarr 组或数据集
    """
    if isinstance(group, zarr.core.Array):        
        # 如果是 agentview_pcd 数据集，进行降采样
        if 'agentview_pcd' in group.name:
            print(f"Performing downsampling on {group.name}")
            points = group[:]  # 原始点云数据 (形状: [B, 128, 128, 6])
            downsampled_points, sampled_indices = downsample_with_fps(points, num_points=1024)

            # 将结果转为 numpy 数组 (形状: [B, 1024, 3])
            downsampled_points_np = np.array(downsampled_points)
            parent_path = group.name.rsplit('/', 1)[0]
            root = zarr.open_group(group.store, mode='r+')
            parent_group = root if parent_path == '' else root[parent_path]
            parent_group.array('pointcloud', downsampled_points_np, overwrite=True)

            print(f"Saved downsampled pointcloud to: {parent_path}/pointcloud")

        else:
            pass

    elif isinstance(group, zarr.Group):
        print(f"Group Name: {group.name}")
        for key in group:
            recursive_read_zarr(group[key])  # 递归读取每个子组或数据集

def read_all_zarr_in_directory(directory):
    """
    读取指定目录下的所有 Zarr 文件夹，并对每个文件夹调用递归读取函数。
    
    :param directory: 存放 Zarr 文件夹的目录路径
    """
    for root, dirs, files in os.walk(directory):
        # 遍历目录中的所有 Zarr 文件夹（目录格式）
        for dir in dirs:
            if dir.endswith('.zarr'):
                zarr_dir_path = os.path.join(root, dir)
                print(f"Opening Zarr folder: {zarr_dir_path}")
                # 打开 Zarr 文件夹并递归读取
                root_group = zarr.open_group(zarr_dir_path, mode='r+')
                recursive_read_zarr(root_group)  # 从根组开始递归读取

# 示例：指定存放 Zarr 文件夹的目录路径
directory_path = '/mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/data/data_libero'
read_all_zarr_in_directory(directory_path)  # 读取目录下所有 Zarr 文件夹
