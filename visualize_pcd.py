import zarr
import numpy as np
from visualizer import visualize_pointcloud

# 读取Zarr文件
zarr_path = "/mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/data/data_maniskill_crop/maniskill_PegInsertionSide-v1_1__expert.zarr"
root = zarr.open(zarr_path, mode='r')
import pdb; pdb.set_trace()
# 获取点云数据（假设存储在'point_cloud'数组中）
# 根据你的数据集结构，可能需要调整索引方式
point_cloud = root['point_cloud'][:]  # 形状通常是 (T, N, 3) 或 (T, N, 6)

# 可视化第一个时间步的点云
# 选择第一个时间步，所有点，前3个坐标（x,y,z）
pc_sample = point_cloud[0, :, :3]  # 形状变为 (N, 3)

# 调用可视化函数
visualize_pointcloud(pc_sample)