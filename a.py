import os
import subprocess
import re

data_directory = "/data1/lxy-24/Data/Maniskill/unimix/WhiteScale___GeneralizedDP3/data/"
result_output_file = "evaluation_summary_1024_wo.txt"

# 正则表达式提取所需指标
pattern_dict = {
    "mean_traj_rewards": r"mean_traj_rewards:\s*([\d.]+)",
    "mean_success_rates": r"mean_success_rates:\s*([\d.]+)",
    "test_mean_score": r"test_mean_score:\s*([\d.]+)",
    "SR_test_L3": r"SR_test_L3:\s*([\d.]+)",
    "SR_test_L5": r"SR_test_L5:\s*([\d.]+)",
}

# 写表头
with open(result_output_file, "w") as summary_file:
    summary_file.write("task_name,command,mean_traj_rewards,mean_success_rates,test_mean_score,SR_test_L3,SR_test_L5\n")

# 遍历并排序 .zarr 文件名
for filename in sorted(os.listdir(data_directory)):
    if filename.endswith(".zarr"):
        # 提取任务名（如 metaworld_sweep_expert.zarr → sweep）
        task_name = filename.split("_")[1]  # 请根据实际格式调整
        print(f"\n[开始处理] 任务: {task_name}")

        # 构造命令字符串
        command = [
            "bash", "scripts/evaluation_policy.sh",
            "dp3", "uni_mix", task_name,
            "gpo-all-0-w", "0", "0"
        ]
        command_str = " ".join(command)
        print(f"[执行命令] {command_str}")

        # 执行命令并捕获输出
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout

        # 提取指标
        extracted_values = [task_name, command_str]
        for key, pattern in pattern_dict.items():
            match = re.search(pattern, output)
            value = match.group(1) if match else "N/A"
            extracted_values.append(value)

        # 输出到终端
        print("[结果]")
        for k, v in zip(pattern_dict.keys(), extracted_values[2:]):
            print(f"  {k}: {v}")

        # 写入结果文件
        with open(result_output_file, "a") as summary_file:
            summary_file.write(",".join(extracted_values) + "\n")

print(f"\n✅ 全部任务处理完成，结果存入：{result_output_file}")
