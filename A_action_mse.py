import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb



import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果用多卡
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积结果一致
    torch.backends.cudnn.benchmark = False     # 固定算法




# 初始化 wandb
wandb.init(project="mlp_256_to_7", name="256_to_7_mlp", config={"learning_rate": 1e-3, "batch_size": 512, "epochs": 100})

# 配置参数
config = wandb.config


#A_single_stablelr_baseline

DATA_PATH = "/A_single_stablelr__query32_real_td_biglr-nce/libero_10/*/demo*.hdf5"

LABEL_PATH =  "/libero/libero_10/*/demo*.hdf5"


BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
LEARNING_RATE = config.learning_rate




def expand_labels(labels, time_steps=7):
    num_samples, label_dim = labels.shape
    expanded_labels = np.zeros((num_samples, (time_steps + 1) * label_dim))

    for i in range(num_samples):
        # 获取当前时间步及后续时间步的标签
        for t in range(time_steps + 1):
            if i + t < num_samples:
                expanded_labels[i, t * label_dim:(t + 1) * label_dim] = labels[i + t]
            else:
                # 超出范围，填充 0
                expanded_labels[i, t * label_dim:(t + 1) * label_dim] = 0

    return expanded_labels




# 自定义数据集类
class HDF5Dataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data_files = sorted(glob.glob(data_path))
        self.label_files = sorted(glob.glob(label_path))
        self.data = []
        self.labels = []

        # 加载数据和标签
        for data_file, label_file in zip(self.data_files, self.label_files):
            with h5py.File(data_file, 'r') as data_h5, h5py.File(label_file, 'r') as label_h5:

          

                

                data = data_h5['root']['actions'][:]

                labels = label_h5['root']['actions'][:]
       
                labels = expand_labels(labels, time_steps=7)

                assert data.shape[0] == labels.shape[0], f"Mismatched T in {data_file}"
                self.data.append(data)
                self.labels.append(labels)

        # 合并所有数据和标签
        self.data = torch.tensor(np.concatenate(self.data, axis=0), dtype=torch.float32)
        self.labels = torch.tensor(np.concatenate(self.labels, axis=0), dtype=torch.float32)
        #breakpoint()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 定义简单的MLP模型
class MLPModel(nn.Module):
    def __init__(self, input_dim=512, output_dim=7):
        super(MLPModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# 训练函数
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)

        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# 测试函数
def test_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)




def inference_and_report(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_errors = []

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            error = outputs - labels  # shape: [batch, 7]
            all_errors.append(error.cpu().numpy())
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)
            total_count += data.size(0)
    
    # 计算整体平均loss
    avg_loss = total_loss / total_count
    # 拼接所有error
    all_errors = np.concatenate(all_errors, axis=0)  # shape: [N, 7]
    # 你可以算MSE, MAE等
    mse = np.mean(all_errors ** 2)
    mae = np.mean(np.abs(all_errors))
    print(f"Inference on all data: MSE={mse:.6f}, MAE={mae:.6f}, Avg Loss={avg_loss:.6f}")
    # 可选：wandb记录
    wandb.log({"final_mse": mse, "final_mae": mae, "final_avg_loss": avg_loss})
    return mse, mae, avg_loss


def main():

    set_seed(42)

    # 初始化设备，强制使用单个 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    train_dataset = HDF5Dataset(DATA_PATH, LABEL_PATH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型、损失函数和优化器 1024 = 32*16*2
    model = MLPModel(input_dim=1024, output_dim=7*8).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 开始训练
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_dataloader, criterion, optimizer, device)
        test_loss = test_model(model, train_dataloader, criterion, device)

        # 打印和记录loss
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss})

    # 保存模型
    torch.save(model.state_dict(), "mlp_256_to_7_model.pth")
    print("Model saved as mlp_256_to_7_model.pth")
    inference_and_report(model, train_dataloader, criterion, device)


if __name__ == "__main__":
    main()