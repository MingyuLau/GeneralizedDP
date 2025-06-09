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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
sys.path.append("/mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy")
from experiments.debug_utils import setup_debug
# import wandb
import cv2
# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from episode_videos.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
# from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_latent_action,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)



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
    task_suite_name: str = "libero_10"               # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
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


from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
def zzget_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": resolution, 
        "camera_widths": resolution,
        "camera_depths": True,
        }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_pcd_from_obs(obs, point_num=10240):
    """
    Extracts point cloud from observation.
    Use real_depth_map, real_rgb_image, camera_intrinsics, 
    camera_extrinsics to get the point cloud.
    
    !!! 仿真中必须加上"camera_depths": True !!!
    """
    depth_map = obs["real_depth_map"]
    rgb_image = obs["agentview_image"]
    camera_intrinsics = obs["camera_intrinsic"]
    camera_extrinsics = obs["camera_extrinsic"]
    import libero.libero.utils.zzutils as zzutils
        
    # 将depth_map转成(H, W)的形状
    depth_map = depth_map.reshape(depth_map.shape[0], depth_map.shape[1])
    
    # 创建一个与depth_map同形状的全False矩阵
    mask = np.zeros_like(depth_map, dtype=bool)

    # 随机选择point_num个索引位置
    flat_indices = np.random.choice(depth_map.size, point_num, replace=False)

    # 将一维索引转换为二维索引
    indices = np.unravel_index(flat_indices, depth_map.shape)

    # 在选定的位置上设置mask为True
    mask[indices] = True

    # 获取True位置的坐标，用于后续点云生成
    mask_coords = np.column_stack(np.where(mask))
    
    pcd = zzutils.depth_map_to_world_point_cloud(depth_map, camera_intrinsics, camera_extrinsics, mask)
    
    # 为pcd添加颜色
    pcd_rgb = rgb_image[mask, :3] 
    pcd = np.hstack((pcd, pcd_rgb))  # Combine point cloud with RGB values
    return pcd

def normalize_point_cloud(point_cloud):
    """
    将点云的xyz坐标均匀归一化在[-1, 1]范围内
    """
    xyz = point_cloud[:, 0:3]
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    
    # 计算归一化比例
    scale = (max_xyz - min_xyz) / 2.0
    center = (max_xyz + min_xyz) / 2.0
    
    # 归一化
    normalized_xyz = (xyz - center) / scale
    
    # 更新点云坐标
    point_cloud[:, 0:3] = normalized_xyz
    return point_cloud
    

import numpy as np
from plyfile import PlyData, PlyElement

# 假设你的点云数据是一个[1024, 6]的numpy数组，命名为point_cloud
# point_cloud[:, 0:3]是xyz坐标，point_cloud[:, 3:6]是rgb颜色值
def downsample_with_fps(points: np.ndarray, num_points: int = 1024):
    """
    使用 farthest point sampling 对点云进行降采样（纯 PyTorch 实现）
    :param points: 输入的点云数据 (numpy 数组), 形状 [N, 6] (xyz+rgb)
    :param num_points: 降采样后的点数
    :return: 降采样后的点云列表和采样点索引列表
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points_tensor = torch.from_numpy(points).float().to(device)
    
    # 分离坐标和颜色信息
    cur_points_xyz = points_tensor[:, :3]  # 只用xyz来计算距离
    cur_points_full = points_tensor  # 保留完整的6通道信息
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
    downsampled_points = [sampled_points_full.cpu().numpy()]
    sampled_indices_list = [indices.cpu().numpy()]
    
    return downsampled_points, sampled_indices_list
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

# 使用示例
# point_cloud = np.random.rand(1024, 6)  # 示例数据
# point_cloud[:, 3:6] *= 255  # 将颜色值缩放到0-255
# save_point_cloud_to_ply(point_cloud, 'point_cloud.ply')

def visualize_3d_trajectory(actions_list, save_path=None, title="Robot Trajectory"):
    """
    可视化机器人在3D空间中的轨迹
    
    Args:
        actions_list: 动作列表，每个动作包含[x, y, z, rx, ry, rz, gripper]
        save_path: 保存图片的路径，如果为None则显示图片
        title: 图片标题
    """
    # 提取xyz坐标
    positions = np.array([[action[0], action[1], action[2]] for action in actions_list])
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹线
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=2, alpha=0.7, label='Trajectory')
    
    # 标记起点和终点
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
              c='green', s=100, label='Start', marker='o')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
              c='red', s=100, label='End', marker='s')
    
    # 绘制中间点
    if len(positions) > 2:
        ax.scatter(positions[1:-1, 0], positions[1:-1, 1], positions[1:-1, 2], 
                  c='blue', s=20, alpha=0.6, label='Waypoints')
    
    # 设置坐标轴标签
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position') 
    ax.set_zlabel('Z Position')  # type: ignore
    ax.set_title(title)
    ax.legend()
    
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
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D轨迹图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_gripper_trajectory(actions_list, save_path=None):
    """
    可视化夹爪状态随时间的变化
    
    Args:
        actions_list: 动作列表，每个动作包含[x, y, z, rx, ry, rz, gripper]
        save_path: 保存图片的路径
    """
    gripper_states = [action[6] for action in actions_list]
    time_steps = range(len(gripper_states))
    
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, gripper_states, 'ro-', linewidth=2, markersize=4)
    plt.xlabel('Time Step')
    plt.ylabel('Gripper State')
    plt.title('Gripper State Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"夹爪状态图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()
    

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)
    # import pdb; pdb.set_trace()
    # Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    # import pdb; pdb.set_trace()
    model = get_model(cfg)

    # wrapped_model Check that the model contains the action un-normalization key
    # if cfg.model_family == "openvla":
    #     # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    #     # with the suffix "_no_noops" in the dataset name)
    #     if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
    #         cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    #     assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # wrapped_model Get Hugging Face processor
    # processor = None
    # if cfg.model_family == "openvla":
    #     processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Tested Ckpt': {cfg.pretrained_checkpoint.split('/')[-1]} \n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    latent_action_detokenize = [f'<ACT_{i}>' for i in range(32)]

    # Start evaluation
    
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = zzget_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            hist_action = ''
            prev_hist_action = ['']

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            replay_pcds = []
            replay_states = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 240  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 300  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 320  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 550  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 420  # longest training demo has 373 steps

            action_queue = deque()
            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            # import pdb; pdb.set_trace()
            while t < max_steps + cfg.num_steps_wait:
                # try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue
                    
                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)
                    pcd = get_pcd_from_obs(obs)
                    # [10240,3] -> [1024, 3]
                    
                    
                    
                    
                    pcd = normalize_point_cloud(pcd)
                    pcd, _ = downsample_with_fps(pcd, 1024)
                    # [1024, 3] -> [1024, 6]
                    pcd = np.hstack((pcd[0], pcd[0]))
                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    replay_pcds.append(pcd)
                    replay_states.append(np.concatenate((obs["robot0_joint_pos"], obs["robot0_gripper_qpos"])))

                    # Prepare observations dict
                    # Note: UniVLA does not take proprio state as input

                    


                    # img_history = replay_images[-cfg.image_history_size:]
                    # if len(img_history) < cfg.image_history_size:
                    #     img_history.extend([replay_images[-1]] * (cfg.image_history_size - len(img_history)))

                    pcd_history = replay_pcds[-cfg.image_history_size:]
                    if len(pcd_history) < cfg.image_history_size:
                        pcd_history.extend([replay_pcds[-1]] * (cfg.image_history_size - len(pcd_history)))

                    state_history = replay_states[-cfg.image_history_size:]
                    if len(state_history) < cfg.image_history_size:
                        state_history.extend([replay_states[-1]] * (cfg.image_history_size - len(state_history)))

                    
                    pcd = np.stack(pcd_history, axis=0)
                    pcd = torch.from_numpy(pcd).unsqueeze(0)
                    # import pdb; pdb.set_trace()
                    # 保存点云到PLY文件（仅保存第一帧用于调试）
                    os.makedirs("/mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/pcd_plots", exist_ok=True)
                    save_point_cloud_to_ply(pcd_history[0], f"/mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/pcd_plots/episode_{total_episodes}_pcd.ply")
                    state = np.stack(state_history, axis=0)
                    state = torch.from_numpy(state).unsqueeze(0)
                    # import pdb; pdb.set_trace()
                    guide_action = torch.zeros(1, 16, 7)
                    # guide_action = torch.ones(1, 16, 7)
                    observation = {
                        "obs": {
                            'point_cloud': pcd,
                            'agent_pos': state,
                        },
                        "actions": guide_action,
                    }
                    
                    # Prepare history latent action tokens
                    # start_
                    # idx = len(prev_hist_action) if len(prev_hist_action) < 4 else 4
                    # prompt_hist_action_list = [prev_hist_action[idx] for idx in range(-1 * start_idx, 0)]
                    # prompt_hist_action = ''
                    # for latent_action in prompt_hist_action_list:
                    #     prompt_hist_action += latent_action
                    
                    # hist_action = ''

                    # Execute action in environment
                    
                    #######################################################
                    ################# Apply ACTIONS here ! ################
                    #######################################################
                    
                    actions = get_action(
                    cfg,
                    model,
                    observation,
                    )
                    # import pdb; pdb.set_trace()
                    # actions = [[0,0,0,0,0,0,0]]
                    # actions = normalize_gripper_action(actions, binarize=True)
                    
                    # 可视化3D轨迹
                    actions_list = actions[0].tolist()
                    trajectory_save_path = f"/mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/trajectory_plots/episode_{total_episodes}_trajectory.png"
                    gripper_save_path = f"/mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/trajectory_plots/episode_{total_episodes}_gripper.png"
                    os.makedirs(os.path.dirname(trajectory_save_path), exist_ok=True)
                    
                    # 绘制3D轨迹
                    visualize_3d_trajectory(actions_list, trajectory_save_path, 
                                          f"Robot 3D Trajectory - Episode {total_episodes}")
                    
                    # 绘制夹爪状态
                    visualize_gripper_trajectory(actions_list, gripper_save_path)
                    
                    print(f"Actions序列长度: {len(actions_list)}")
                    print(f"第一个动作: {actions_list[0]}")
                    print(f"最后一个动作: {actions_list[-1]}")
                    
                    video_path = f"/mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/episode_videos/episode_{total_episodes}_zeros.mp4"
                    os.makedirs(os.path.dirname(video_path), exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 30  # 根据实际需要调整帧率
                    img_height, img_width = img.shape[:2]  # 假设img已经定义
                    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (img_width, img_height))
                    
                    for idx, action in enumerate(actions[0].tolist()):
                        action = np.array(action)
                        obs, reward, done, info = env.step(action)
                        img = get_libero_image(obs, resize_size)
                        video_writer.write(img)
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1
                    video_writer.release()
                    # import pdb; pdb.set_trace()
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter  
                    # action = [0, 0, 0, 0, 0, 0, 0]
                    # obs, reward, done, info = env.step(action)
                    # import pdb; pdb.set_trace()

                    
                    #######################################################
                    ################# Apply ACTIONS here ! ################
                    #######################################################
                    
                    if done:
                        break

                # except Exception as e:
                #     print(f"Caught exception: {e}")
                #     log_file.write(f"Caught exception: {e}\n")
                #     break

            task_episodes += 1
            total_episodes += 1

            if cfg.save_video:
                # Save a replay video of the episode
                save_rollout_video(
                    replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    # Save local log file
    log_file.close()





# import debugpy
# debugpy.listen(("0.0.0.0", 15092)) 
# print("⏳ Waiting for debugger attach...")
# debugpy.wait_for_client()
# print("✅ Debugger attached.")


if __name__ == "__main__":
    # setup_debug(True)
    eval_libero()
