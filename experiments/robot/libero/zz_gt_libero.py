



from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import (GroupedTaskDataset, SequenceVLDataset, get_dataset)
from libero.libero import benchmark, get_libero_path
import os
import torch



benchmark_name = "libero_10" # can be from {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
dataset_path = "/mnt/hwfile/3dv/liumingyu/zzdata/LIBERO/LIBERO-datasets"

obs_modality = {'rgb': ['agentview_rgb', 'eye_in_hand_rgb'], 'depth': [], 'low_dim': ['gripper_states', 'joint_states']}
seq_len = 10


benchmark = get_benchmark(benchmark_name)(0)

# prepare datasets from the benchmark
datasets = []
descriptions = []
shape_meta = None
n_tasks = benchmark.n_tasks

for i in range(n_tasks):
    # currently we assume tasks from same benchmark have the same shape_meta
    task_i_dataset, shape_meta = get_dataset(
            dataset_path=os.path.join(dataset_path, benchmark.get_task_demonstration(i)),
            obs_modality=obs_modality,
            initialize_obs_utils=(i==0),
            seq_len=seq_len,
    )
    # add language to the vision dataset, hence we call vl_dataset
    descriptions.append(benchmark.get_task(i).language)
    datasets.append(task_i_dataset)

task_embs = torch.zeros(n_tasks, 512)

benchmark.set_task_embs(task_embs)

datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(datasets, task_embs)]
n_demos = [data.n_demos for data in datasets]
n_sequences = [data.total_num_sequences for data in datasets]




def get_actions(datasets, task_id, step, target_demo_id):
    """
    向量化实现：获取指定任务和时间步的动作。

    参数:
        datasets (list): 数据集列表。
        task_id (int): 当前任务的 ID。
        step (int): 当前时间步。

    返回:
        np.array: 动作数组，形状为 (n_demos, 7)。
    """
    import numpy as np

    # 获取任务对应的数据集
    hdf5_cache = datasets[task_id].sequence_dataset.hdf5_cache
    sequence_dataset = datasets[task_id].sequence_dataset

    # 获取所有演示的 ID 和对应的长度
    demo_ids = np.array(sequence_dataset.demos)
    demo_lengths = np.array([sequence_dataset._demo_id_to_demo_length[demo_id] for demo_id in demo_ids])

    if f"demo_{target_demo_id}" not in demo_ids:
        raise ValueError(f"Target demo ID {target_demo_id} not found in the dataset.")
    
    if step+16 >= demo_lengths[target_demo_id]:
        raise ValueError(f"Step {step} exceeds the length of the demonstration {target_demo_id}.")

    return hdf5_cache[f"demo_{target_demo_id}"]["actions"][step: step+16]

