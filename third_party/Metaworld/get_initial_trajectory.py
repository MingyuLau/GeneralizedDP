#<lxy>
import argparse
import os
import zarr
import numpy as np
from diffusion_policy_3d.env import MetaWorldEnv
from termcolor import cprint
import copy
import imageio
from metaworld.policies import *


def load_mw_policy(task_name):
    if task_name == 'peg-insert-side':
        agent = SawyerPegInsertionSideV2Policy()
    else:
        task_name = task_name.split('-')
        task_name = [s.capitalize() for s in task_name]
        task_name = "Sawyer" + "".join(task_name) + "V2Policy"
        agent = eval(task_name)()
    return agent

def main(args):
    env_name = args.env_name

    save_dir = os.path.join(args.root_dir, 'metaworld_' + args.env_name + 'test_traj.zarr')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    e = MetaWorldEnv(env_name, device="cuda:0", use_point_crop=True)
    test_traj_num = args.test_traj_num

    #只需要存储轨迹位置
    state_arrays = []
    seed_arrays = []

    current_traj_idx = 0
    mw_policy = load_mw_policy(env_name)
    while current_traj_idx < test_traj_num:
        np.random.seed(args.seed_base+current_traj_idx)
        raw_state = e.reset()
        obs_dict = e.get_visual_obs()
        done = False
        current_traj_idx += 1

        state_array_per_traj = []
        while not done:
            action = mw_policy.get_action(raw_state)
            obs_dict, reward, done, info = e.step(action)
            state_array_per_traj.append(obs_dict['full_state'])
            raw_state = obs_dict['full_state']

            ep_reward += reward
            ep_success = ep_success or info['success']
			ep_success_times += info['success']
            if done:
                break
        
        if not ep_success or ep_success_times < 5:
            print("Episode {}: success = {}, success times = {}".format(current_traj_idx, ep_success, ep_success_times))
            continue
        else:
             state_arrays.append(copy.deepcopy(state_array_per_traj))
             seed_arrays.append(args.seed_base+current_traj_idx)

    # save data        
    zarr_root = zarr.group(save_dir)
	zarr_data = zarr_root.create_group('data')
	zarr_meta = zarr_root.create_group('meta')
	compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    zarr_data.create_dataset('state', data=state_arrays, compressor=compressor)
    zarr_meta.create_dataset('seed', data=seed_arrays, compressor=compressor)

    print("Data saved to {}".format(save_dir))
    print("Total {} trajectories".format(len(state_arrays)))
    print("Total {} seeds".format(len(seed_arrays)))
    del state_arrays, seed_arrays
    del zarr_data, zarr_meta
    del e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='door-open-v2')
    parser.add_argument('--root_dir', type=str, default='/home/yangchen/yangchen/diffusion_policy_3d')
    parser.add_argument('--test_traj_num', type=int, default=100)
    parser.add_argument('--seed_base', type=int, default=0)
    args = parser.parse_args()

    main(args)
#</lxy>