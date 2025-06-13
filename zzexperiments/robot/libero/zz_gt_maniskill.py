
import h5py
import numpy as np

def read_hdf5_group(file_path, group_name):
    def read_group(group):
        group_data = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                # 如果是子组，递归读取
                group_data[key] = read_group(item)
            elif isinstance(item, h5py.Dataset):
                # 如果是数据集，读取数据
                group_data[key] = item[:]
        return group_data

    with h5py.File(file_path, 'r') as f:
        if group_name in f:
            group = f[group_name]
            return read_group(group)
        else:
            print(f"Group '{group_name}' not found in the file.")
            return None

path = "/home/hz/code/PointCloudMatters/data/maniskill2/demos/v0/rigid_body/StackCube-v0/trajectory.rgbd.pd_ee_delta_pose.h5"











