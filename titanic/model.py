import torch.nn as nn
import torch

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # nn.Linear也继承自nn.Module，输入为input_dim,输出一个值

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Logistic Regression 输出概率