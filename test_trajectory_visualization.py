import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_3d_trajectory(actions_tensor, save_path=None, title="Robot Trajectory"):
    """
    可视化机器人在3D空间中的轨迹
    
    Args:
        actions_tensor: 动作tensor，形状为[N, 7]，包含[x, y, z, rx, ry, rz, gripper]
        save_path: 保存图片的路径，如果为None则显示图片
        title: 图片标题
    """
    # 转换为numpy数组
    if isinstance(actions_tensor, torch.Tensor):
        actions_array = actions_tensor.cpu().numpy()
    else:
        actions_array = actions_tensor
    
    # 提取xyz坐标
    positions = actions_array[:, :3]  # [x, y, z]
    
    # 创建3D图
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹线
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=3, alpha=0.8, label='Trajectory')
    
    # 标记起点和终点
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
              c='green', s=200, label='Start', marker='o', edgecolor='black', linewidth=2)
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
              c='red', s=200, label='End', marker='s', edgecolor='black', linewidth=2)
    
    # 绘制中间点，根据夹爪状态着色
    gripper_states = actions_array[:, 6]  # 夹爪状态
    for i in range(1, len(positions)-1):
        color = 'orange' if gripper_states[i] > 0 else 'purple'  # 开启为橙色，关闭为紫色
        ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], 
                  c=color, s=50, alpha=0.7)
    
    # 添加夹爪状态图例
    ax.scatter([], [], [], c='orange', s=50, label='Gripper Open')
    ax.scatter([], [], [], c='purple', s=50, label='Gripper Closed')
    
    # 设置坐标轴标签
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12) 
    ax.set_zlabel('Z Position', fontsize=12)  # type: ignore
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    # 设置坐标轴比例相等
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                         positions[:, 1].max()-positions[:, 1].min(),
                         positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # type: ignore
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D轨迹图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_trajectory_stats(actions_tensor):
    """
    分析轨迹统计信息
    """
    if isinstance(actions_tensor, torch.Tensor):
        actions_array = actions_tensor.cpu().numpy()
    else:
        actions_array = actions_tensor
    
    positions = actions_array[:, :3]
    orientations = actions_array[:, 3:6]
    gripper_states = actions_array[:, 6]
    
    print("=== 轨迹统计信息 ===")
    print(f"轨迹点数量: {len(actions_array)}")
    print(f"位置范围:")
    print(f"  X: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}]")
    print(f"  Y: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}]")
    print(f"  Z: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}]")
    
    print(f"姿态范围:")
    print(f"  RX: [{orientations[:, 0].min():.4f}, {orientations[:, 0].max():.4f}]")
    print(f"  RY: [{orientations[:, 1].min():.4f}, {orientations[:, 1].max():.4f}]")
    print(f"  RZ: [{orientations[:, 2].min():.4f}, {orientations[:, 2].max():.4f}]")
    
    print(f"夹爪状态:")
    open_count = np.sum(gripper_states > 0)
    closed_count = np.sum(gripper_states < 0)
    print(f"  开启状态: {open_count} 步")
    print(f"  关闭状态: {closed_count} 步")
    
    # 计算轨迹总长度
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_distance = np.sum(distances)
    print(f"轨迹总长度: {total_distance:.4f}")
    print(f"平均步长: {np.mean(distances):.4f}")

def plot_trajectory_components(actions_tensor, save_path=None):
    """
    绘制轨迹各个分量随时间的变化
    """
    if isinstance(actions_tensor, torch.Tensor):
        actions_array = actions_tensor.cpu().numpy()
    else:
        actions_array = actions_tensor
    
    time_steps = np.arange(len(actions_array))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 位置分量
    axes[0].plot(time_steps, actions_array[:, 0], 'r-', label='X', linewidth=2)
    axes[0].plot(time_steps, actions_array[:, 1], 'g-', label='Y', linewidth=2)
    axes[0].plot(time_steps, actions_array[:, 2], 'b-', label='Z', linewidth=2)
    axes[0].set_title('Position Components', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Position')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 姿态分量
    axes[1].plot(time_steps, actions_array[:, 3], 'r--', label='RX', linewidth=2)
    axes[1].plot(time_steps, actions_array[:, 4], 'g--', label='RY', linewidth=2)
    axes[1].plot(time_steps, actions_array[:, 5], 'b--', label='RZ', linewidth=2)
    axes[1].set_title('Orientation Components', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Orientation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 夹爪状态
    axes[2].plot(time_steps, actions_array[:, 6], 'ko-', linewidth=2, markersize=4)
    axes[2].set_title('Gripper State', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Gripper State')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-1.5, 1.5])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分量图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    # 您提供的tensor数据
    actions_tensor = torch.tensor([
        [-0.0736, -0.0330,  0.4857,  0.1777,  0.4150, -0.5275,  1.0000],
        [-0.1319, -0.0450,  0.2057,  0.2613,  0.5300, -0.5081,  1.0000],
        [-0.1963, -0.0511,  0.0143,  0.3380,  0.6300, -0.5016,  1.0000],
        [-0.2914, -0.0511, -0.0257,  0.3868,  0.6100, -0.5210,  1.0000],
        [-0.3374, -0.0511, -0.1114,  0.3868,  0.5450, -0.5469, -1.0000],
        [-0.4724, -0.0511, -0.4743,  0.2544,  0.3650, -0.5469, -1.0000],
        [-0.5123, -0.0511, -0.6400,  0.2056,  0.2900, -0.4822, -1.0000],
        [-0.5092, -0.0511, -0.6971,  0.1568,  0.3050, -0.4498, -1.0000],
        [-0.4816, -0.0511, -0.7486,  0.1847,  0.3600, -0.3333, -1.0000],
        [-0.4571, -0.0270, -0.7514,  0.2125,  0.3800, -0.2233, -1.0000],
        [-0.4479,  0.0721, -0.7086,  0.1777,  0.3250, -0.2233, -1.0000],
        [-0.4479,  0.1261, -0.6657,  0.1568,  0.2350, -0.2233, -1.0000],
        [-0.3896,  0.1802, -0.5971,  0.1080,  0.1000, -0.2233, -1.0000],
        [-0.3221,  0.1111, -0.5686,  0.1638,  0.1000, -0.2233, -1.0000],
        [-0.3252, -0.0330, -0.5743,  0.2613,  0.1000, -0.2233, -1.0000],
        [-0.3834, -0.0811, -0.5743,  0.3171,  0.1000, -0.4369, -1.0000]
    ], device='cuda:0')
    
    print("开始分析机器人轨迹数据...")
    
    # 分析统计信息
    analyze_trajectory_stats(actions_tensor)
    print("\n" + "="*50 + "\n")
    
    # 创建保存目录
    save_dir = "./trajectory_analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    # 3D轨迹可视化
    print("生成3D轨迹可视化...")
    visualize_3d_trajectory(
        actions_tensor, 
        save_path=f"{save_dir}/3d_trajectory.png",
        title="Robot 3D Trajectory Analysis"
    )
    
    # 分量图
    print("生成轨迹分量图...")
    plot_trajectory_components(
        actions_tensor,
        save_path=f"{save_dir}/trajectory_components.png"
    )
    
    print(f"\n所有可视化图片已保存到: {save_dir}/")

if __name__ == "__main__":
    main() 