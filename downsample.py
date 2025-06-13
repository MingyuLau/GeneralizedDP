import os
import zarr
import numpy as np
import pytorch3d.ops as torch3d_ops
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3DType
import random
import time

# 点云降采样函数（纯PyTorch实现）
def downsample_with_color_fps(points: np.ndarray, num_points: int = 1024):
    """
    使用 farthest point sampling 对点云进行降采样（纯 PyTorch 实现）
    :param points: 输入的点云数据 (numpy 数组), 形状 [B, H, W, 6] (xyz+rgb)
    :param num_points: 降采样后的点数
    :return: 降采样后的点云列表和采样点索引列表
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points_tensor = torch.from_numpy(points).float().to(device)
    batch_size = points_tensor.shape[0]
    
    downsampled_points = []
    sampled_indices_list = []
    
    for i in range(batch_size):
        # 提取当前批次的点云并展平 [H*W, 6]
        cur_points_full = points_tensor[i, :, :, :].reshape(-1, 6)  # 保留完整的6通道信息
        cur_points_xyz = cur_points_full[:, :3]  # 只用xyz来计算距离
        total_points = cur_points_xyz.shape[0]
        
        if num_points > total_points:
            raise ValueError(f"num_points ({num_points}) > total points ({total_points})")
        
        # 初始化采样索引和距离数组
        indices = torch.zeros(num_points, dtype=torch.long, device=device)
        distances = torch.full((total_points,), float('inf'), device=device)
        
        # 随机选择第一个点
        farthest_idx = torch.randint(0, total_points, (1,), device=device)
        indices[0] = farthest_idx
        
        # 迭代选择剩余点
        for j in range(1, num_points):
            # 计算最新采样点到所有点的距离（只用xyz坐标）
            new_point = cur_points_xyz[indices[j-1]]
            dist_to_new = torch.norm(cur_points_xyz - new_point, dim=1)
            
            # 更新最小距离
            distances = torch.min(distances, dist_to_new)
            
            # 选择距离最大的点作为下一个采样点
            farthest_idx = torch.argmax(distances)
            indices[j] = farthest_idx
        
        # 收集采样点（保留完整的6个通道：xyz+rgb）
        sampled_points_full = cur_points_full[indices]
        downsampled_points.append(sampled_points_full.cpu().numpy())
        sampled_indices_list.append(indices.cpu().numpy())
    
    return downsampled_points, sampled_indices_list

def downsample_with_fps(points: np.ndarray, num_points: int = 1024):
    """
    使用 farthest point sampling 对点云进行降采样（纯 PyTorch 实现）
    :param points: 输入的点云数据 (numpy 数组), 形状 [B, H, W, C]
    :param num_points: 降采样后的点数
    :return: 降采样后的点云列表和采样点索引列表
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points_tensor = torch.from_numpy(points).float().to(device)
    batch_size = points_tensor.shape[0]
    
    downsampled_points = []
    sampled_indices_list = []
    
    for i in range(batch_size):
        # 提取当前批次的点云并展平 [H*W, 3]
        cur_points = points_tensor[i, :, :, :3].reshape(-1, 3)
        total_points = cur_points.shape[0]
        
        if num_points > total_points:
            raise ValueError(f"num_points ({num_points}) > total points ({total_points})")
        
        # 初始化采样索引和距离数组
        indices = torch.zeros(num_points, dtype=torch.long, device=device)
        distances = torch.full((total_points,), float('inf'), device=device)
        
        # 随机选择第一个点
        farthest_idx = torch.randint(0, total_points, (1,), device=device)
        indices[0] = farthest_idx
        
        # 迭代选择剩余点
        for j in range(1, num_points):
            # 计算最新采样点到所有点的距离
            new_point = cur_points[indices[j-1]]
            dist_to_new = torch.norm(cur_points - new_point, dim=1)
            
            # 更新最小距离
            distances = torch.min(distances, dist_to_new)
            
            # 选择距离最大的点作为下一个采样点
            farthest_idx = torch.argmax(distances)
            indices[j] = farthest_idx
        
        # 收集采样点
        sampled_points = cur_points[indices]
        downsampled_points.append(sampled_points.cpu().numpy())
        sampled_indices_list.append(indices.cpu().numpy())
    
    return downsampled_points, sampled_indices_list

# 点云降采样函数（使用pytorch3d）
def downsample_with_fps1(points: np.ndarray, num_points: int = 1024):
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

# 可视化函数
def visualize_pointcloud_comparison(original_points, downsampled_points, save_path=None):
    """
    可视化原始点云和降采样点云的对比
    
    :param original_points: 原始点云数据 (形状: [128, 128, 3] 或 [16384, 3])
    :param downsampled_points: 降采样后的点云 (形状: [1024, 3])
    :param save_path: 图像保存路径（如果为None则显示）
    """
    # 确保输入点云是正确形状
    if original_points.ndim == 3:
        original_points = original_points.reshape(-1, 3)
    
    # 随机采样部分原始点以避免过于密集
    if original_points.shape[0] > 5000:
        indices = np.random.choice(original_points.shape[0], 5000, replace=False)
        sampled_original = original_points[indices]
    else:
        sampled_original = original_points
    
    # 创建3D可视化
    fig = plt.figure(figsize=(16, 8))
    
    # 原始点云可视化
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(sampled_original[:, 0], sampled_original[:, 1], sampled_original[:, 2], 
               s=2, c='b', alpha=0.5)
    ax1.set_title(f'原始点云 ({original_points.shape[0]}点)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    try:
        ax1.set_zlabel('Z')
    except AttributeError:
        pass
    
    # 设置相同的坐标范围
    all_xyz = np.vstack([original_points, downsampled_points])
    min_vals = np.min(all_xyz, axis=0)
    max_vals = np.max(all_xyz, axis=0)
    
    ax1.set_xlim(min_vals[0], max_vals[0])
    ax1.set_ylim(min_vals[1], max_vals[1])
    try:
        ax1.set_zlim(min_vals[2], max_vals[2])
    except AttributeError:
        pass
    
    # 降采样点云可视化
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(downsampled_points[:, 0], downsampled_points[:, 1], downsampled_points[:, 2], 
               s=10, c='r', alpha=0.8)
    ax2.set_title(f'降采样点云 ({downsampled_points.shape[0]}点)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    try:
        ax2.set_zlabel('Z')
    except AttributeError:
        pass
    ax2.set_xlim(min_vals[0], max_vals[0])
    ax2.set_ylim(min_vals[1], max_vals[1])
    try:
        ax2.set_zlim(min_vals[2], max_vals[2])
    except AttributeError:
        pass
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"可视化结果已保存至: {save_path}")
    else:
        plt.show()

# 修改后的递归读取函数
def recursive_read_zarr(group, visualize=False, vis_dir=None):
    """
    递归读取 Zarr 数据，并进行降采样处理
    
    :param group: 当前的 Zarr 组或数据集
    :param visualize: 是否进行可视化
    :param vis_dir: 可视化图像保存目录
    """
    if isinstance(group, zarr.Array):        
        # 如果是 agentview_pcd 数据集，进行降采样
        if hasattr(group, 'name') and group.name and 'agentview_pcd' in group.name:
            print(f"Performing downsampling on {group.name}")
            points = group[:]  # 原始点云数据 (形状: [B, 128, 128, 6])
            
            # 进行降采样
            downsampled_points, sampled_indices = downsample_with_color_fps(points, num_points=1024)
            
            # 将结果转为 numpy 数组 (形状: [B, 1024, 6])
            downsampled_points_np = np.array(downsampled_points)
            if hasattr(group, 'name') and group.name:
                parent_path = group.name.rsplit('/', 2)[0]
                root = zarr.open_group(group.store, mode='r+')
                parent_group = root if parent_path == '' else root[parent_path]
                
                # 保存降采样后的点云
                try:
                    parent_group.array('pointcloud', downsampled_points_np, overwrite=True)
                    print(f"Saved downsampled pointcloud to: {parent_path}/pointcloud, {downsampled_points_np.shape}")
                except Exception as e:
                    print(f"Failed to save downsampled pointcloud: {e}")

    elif isinstance(group, zarr.Group):
        if hasattr(group, 'name'):
            print(f"Group Name: {group.name}")
        for key in group:
            recursive_read_zarr(group[key], visualize, vis_dir)  # 递归读取每个子组或数据集

# 修改后的目录读取函数
def read_all_zarr_in_directory(directory, visualize=False, vis_dir=None):
    """
    读取指定目录下的所有 Zarr 文件夹，并对每个文件夹调用递归读取函数
    
    :param directory: 存放 Zarr 文件夹的目录路径
    :param visualize: 是否进行可视化
    :param vis_dir: 可视化图像保存目录
    """
    # 创建可视化目录
    if visualize and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        print(f"可视化图像将保存至: {vis_dir}")
    
    for root, dirs, files in os.walk(directory):
        # 遍历目录中的所有 Zarr 文件夹（目录格式）
        for dir_name in dirs:
            if dir_name.endswith('.zarr'):
                zarr_dir_path = os.path.join(root, dir_name)
                print(f"\n{'='*50}")
                print(f"Processing Zarr folder: {zarr_dir_path}")
                print(f"{'='*50}")
                
                # 打开 Zarr 文件夹并递归读取
                root_group = zarr.open_group(zarr_dir_path, mode='r+')
                recursive_read_zarr(root_group, visualize, vis_dir)  # 从根组开始递归读取
            else:
                print(f"Skipping non-zarr directory: {dir_name}")

# 主程序
if __name__ == "__main__":
    # 指定存放 Zarr 文件夹的目录路径
    directory_path = '/mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/data/data_libero10'
    
    # 可视化设置
    VISUALIZE = True  # 设置为True启用可视化
    VISUALIZATION_DIR = "./fps_visualizations"  # 可视化图像保存目录
    
    # 读取目录下所有 Zarr 文件夹
    read_all_zarr_in_directory(
        directory_path, 
        visualize=VISUALIZE, 
        vis_dir=VISUALIZATION_DIR
    )