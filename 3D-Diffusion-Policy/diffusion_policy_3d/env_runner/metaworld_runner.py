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

    def run(self, policy: BasePolicy, save_video=False):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    print(obs_dict_input['agent_pos'].shape)
                    # for key in obs_dict.keys():
                    #     print(f"key: {key}, obs_dict[key].shape: {obs_dict[key].shape}")
                    if self.use_sparse_action:
                        sparse_action0=self.mw_policy.get_action(obs_dict['full_state'][0].squeeze().cpu().numpy())
                        sparse_action1=self.mw_policy.get_action(obs_dict['full_state'][1].squeeze().cpu().numpy())
                        sparse_action= np.array([sparse_action0,sparse_action1])
                        sparse_action = sparse_action.reshape(1, 2, 4)
                        #print(f"sparse_action.shape: {sparse_action.shape}, obs_dict['full_state'].shape: {obs_dict['full_state'].shape}")
                        action_dict = policy.predict_action_w_sparse(obs_dict_input,sparse_action)
                    else:
                        action_dict = policy.predict_action(obs_dict_input)
                    

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                print(action.shape)

                obs, reward, done, info = env.step(action)
           

                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        if save_video:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        return log_data
