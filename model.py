import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class LIFNeuron(nn.Module):
    def __init__(self, input_size, tau=2.0, v_threshold=1.0, v_reset=0.0):
        super(LIFNeuron, self).__init__()
        self.input_size = input_size
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v = torch.zeros(input_size)

    def forward(self, x):
        # 避免 inplace 操作，避免修改 self.v
        self.v = self.v + (1 / self.tau) * (-self.v + x)  # 修改为非 in-place 操作
        spike = (self.v >= self.v_threshold).float()
        self.v[spike == 1] = self.v_reset
        return self.v, spike


class Spiking_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Spiking_QNet, self).__init__()
        # 输入层到隐藏层的突触
        self.fc1 = nn.Linear(input_size, hidden_size)
        # LIF 神经元层
        self.lif1 = LIFNeuron(hidden_size)
        # 隐藏层到输出层的突触
        self.fc2 = nn.Linear(hidden_size, output_size)
        # 输出层使用非脉冲神经元（LI）
        self.li = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 突触层计算
        x = F.relu(self.fc1(x))
        # 使用LIF神经元
        v, spike = self.lif1(x)
        # 使用LI模型来输出Q值
        output = self.li(v)
        return output

    def save(self, file_name='model.pth'):
        """
        Save the model state to a file.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # 将状态和下一状态转换为 tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # 将状态转换为 (1, x) 的形状以适配模型输入
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 预测当前状态的 Q 值
        pred = self.model(state)  # 输出形状: (batch_size, num_actions)
        target = pred.clone()  # 克隆预测值

        # 确保 target 是二维的，如果是单个样本，增加一个维度
        if len(target.shape) == 1:
            target = target.unsqueeze(0)

        # 更新 Q 值
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # 找到 action 中值为 1 的索引，表示执行的动作
            action_idx = torch.argmax(action[idx]).item()
            if action_idx < target.size(1):  # 检查动作索引是否超出 target 的范围
                target[idx][action_idx] = Q_new
            else:
                raise ValueError(f"Action index {action_idx} is out of bounds for target with size {target.size(1)}")

        # 梯度清零
        self.optimizer.zero_grad()
        # 计算损失
        loss = self.criterion(target, pred)
        loss.backward()
        # 更新模型参数
        self.optimizer.step()


