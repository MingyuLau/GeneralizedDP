import wandb
import numpy as np
import torch
import collections
import tqdm
import zarr
import os

from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint



import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'third_party')))
print("sys_path",sys.path)
from Metaworld.metaworld.policies import *
def load_mw_policy(task_name):
    if task_name == 'peg-insert-side':
        agent = SawyerPegInsertionSideV2Policy()
    else:
        task_name = task_name.split('-')
        task_name = [s.capitalize() for s in task_name]
        task_name = "Sawyer" + "".join(task_name) + "V2Policy"
        agent = eval(task_name)()
    return agent

class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:1",
                 use_point_crop=True,
                 num_points=512,
                 use_sparse_action=True,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        self.output_dir = output_dir
        print(f"output_dir: {self.output_dir},{output_dir}")
        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
                
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)
        self.env_expert= MetaWorldEnv(self.task_name, device="cuda:0", use_point_crop=True)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.use_sparse_action = use_sparse_action
        if use_sparse_action:
            self.mw_policy = load_mw_policy(task_name)
        # self.expert_trajectorys=self.load_expert_trajectory()

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)


    # def load_expert_trajectory(self):
    #     # 打开 Zarr 文件

    #     traj_path= os.path.join('/home/lxy-24/workspace/GeneralizedDP/3D-Diffusion-Policy/test_data/metaworld_' + self.task_name + 'test_traj.zarr')
    #     zarr_root = zarr.open(traj_path, mode='r')
        
    #     # 假设 trajectory 数据保存在 'data' group 下
    #     zarr_data = zarr_root['data']
    #     # 加载具体字段，例如 observations, actions, rewards 等
    #     observations = zarr_data['state'][:]
    #     gt_actions = zarr_data["action"][:]
    #     print(f"observations: {observations.shape}, gt_actions: {gt_actions.shape}")

    #     # 可以进一步封装成字典返回
    #     return {
    #         'observations': observations,
    #         'actions': gt_actions,
    #     }

    def student_eval(self):
        pass
    
    def expert_traj(self, env, policy, device):
        obs = env.reset()
        done = False
        traj_reward = 0
        is_success = False
        expert_traj = []

        # 辅助函数：安全转换NumPy数组为PyTorch张量
        def numpy_to_tensor(x):
            if isinstance(x, np.ndarray):
                # 确保数组是连续内存且可写
                x = np.ascontiguousarray(x.copy())
            return torch.from_numpy(x).to(device=device) if isinstance(x, np.ndarray) else x

        # 收集专家轨迹
        while not done:
            # 1. 处理观测数据
            np_obs_dict = dict(obs)

            # 2. 获取策略动作
            with torch.no_grad():

                # 获取动作并确保是NumPy数组
                action_np = self.mw_policy.get_action(
                    np_obs_dict['full_state']
                )
                
                # 验证动作格式
                if isinstance(action_np, torch.Tensor):
                    action_np = action_np.cpu().numpy()
                action_np = np.asarray(action_np, dtype=np.float32).flatten()  # 确保是一维数组


            # 3. 执行环境步骤
            try:
                obs, reward, done, info = env.step(action_np)
                
                # 处理done标志（适应gym和metaworld的不同格式）
                done = bool(done) if isinstance(done, (bool, np.bool_)) else np.all(done)
                
                # 记录轨迹
                expert_traj.append({
                    'obs': np_obs_dict,
                    'action': action_np,
                    'reward': reward,
                    'done': done,
                    'info': info
                })
                
                traj_reward += reward
                if 'success' in info:
                    is_success = info['success']
                    
            except Exception as e:
                print(f"Error executing step with action: {action_np}")
                print(f"Action type: {type(action_np)}, shape: {action_np.shape}")
                print(f"Observation keys: {obs.keys() if isinstance(obs, dict) else 'N/A'}")
                raise e

        # 返回轨迹信息
        return {
            'trajectory': expert_traj,
            'total_reward': traj_reward,
            'is_success': is_success
        }


    def run(self, policy: BasePolicy, save_video=False):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env
        env_exp = self.env_expert

        # 预加载专家轨迹
        export_traj = self.expert_traj(env_exp, self.mw_policy, device)
        if not isinstance(export_traj['trajectory'], (list, np.ndarray)) or len(export_traj['trajectory']) == 0:
            raise ValueError("Invalid expert trajectory data")

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), 
                                    desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", 
                                    leave=False, 
                                    mininterval=self.tqdm_interval_sec):
            
            # 初始化环境
            obs = env.reset()
            policy.reset()
            done = False
            traj_reward = 0
            is_success = False
            current_step = 0

            while not done and current_step < len(export_traj['trajectory']):
                # 准备观测数据
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                    lambda x: torch.from_numpy(x).to(device=device))
                
                # 构建策略输入
                obs_dict_input = {
                    'obs': {
                        'point_cloud': obs_dict['point_cloud'].unsqueeze(0),
                        'agent_pos': obs_dict['agent_pos'].unsqueeze(0)
                    }
                }
                with torch.no_grad():
                    if self.use_sparse_action:
                        # 安全获取专家轨迹数据
                        if current_step*8 + 4 > len(export_traj['trajectory']):
                            break  # 避免越界
                            
                        frame0 = export_traj['trajectory'][current_step*8]['obs']['full_state'][:4]
                        frame1 = export_traj['trajectory'][current_step*8+4]['obs']['full_state'][:4]
                        # 直接从专家轨迹获取动作
                        sparse_action= np.array([frame0,frame1])
                        print(sparse_action.shape)
                        sparse_action = sparse_action.reshape(1, 2, 4)
                        obs_dict_input['actions']=sparse_action
                        # 常规策略预测
                        action_dict = policy.predict_action(obs_dict_input)
                        np_action_dict = dict_apply(action_dict,
                                                    lambda x: x.detach().to('cpu').numpy())
                        action = np_action_dict['action'].squeeze(0)
                        

                    # 验证动作格式
                    if not isinstance(action, np.ndarray):
                        action = np.array(action, dtype=np.float32)
                    print(action.shape)
                    # action = action.flatten()  # 确保是一维数组

                    # 执行环境步骤
                    obs, reward, done, info = env.step(action)
                    current_step += 1
                    traj_reward += reward
                    is_success = is_success or max(info['success'])

            # 记录结果
            all_success_rates.append(float(is_success))
            all_traj_rewards.append(traj_reward)

        # 计算结果指标
        log_data = {
            'mean_traj_rewards': np.mean(all_traj_rewards),
            'mean_success_rates': np.mean(all_success_rates),
            'test_mean_score': np.mean(all_success_rates),
            # 'SR_test_L3': self.logger_util_test.average_of_largest_K(),
            # 'SR_test_L5': self.logger_util_test10.average_of_largest_K()
        }

        cprint(f"test_mean_score: {log_data['test_mean_score']}", 'green')

        # 处理视频
        if save_video:
            videos = env.env.get_video()
            if len(videos.shape) == 5:
                videos = videos[:, 0]  # select first frame
            log_data['sim_video_eval'] = wandb.Video(videos, fps=self.fps, format="mp4")

        return log_data