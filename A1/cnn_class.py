import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBase(nn.Module):
  def __init__(self):
    super().__init__()
    # conv base:
    self.conv32 = nn.Conv2d(1, 32, 3, padding=1)
    self.conv64 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv128_1 = nn.Conv2d(64, 128, 3, padding=1)
    self.conv128_2 = nn.Conv2d(128, 128, 3, padding=1)
    self.conv256_1 = nn.Conv2d(128, 256, 3, padding=1)
    self.conv256_2 = nn.Conv2d(256, 256, 3, padding=1)
    self.bn32 = nn.BatchNorm2d(32)
    self.bn64 = nn.BatchNorm2d(64)
    self.bn128_1 = nn.BatchNorm2d(128)
    self.bn128_2 = nn.BatchNorm2d(128)
    self.bn256_1 = nn.BatchNorm2d(256)
    self.bn256_2 = nn.BatchNorm2d(256)
    self.pool = nn.MaxPool2d(2, 2)
    # FC:
    self.fc1 = nn.Linear(9*9*256, 512)
    self.fc2 = nn.Linear(512, 128)


class OneHeadedCNN(CNNBase):
    def __init__(self, out_dim):
        super().__init__()
        self.head = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn32(self.conv32(x))))
        # print(f'Block C1 mean: {x.mean().item():.4f}, std:{x.std().item():.4f}')
        x = self.pool(F.relu(self.bn64(self.conv64(x))))
        # print(f'Block C2 mean: {x.mean().item():.4f}, std:{x.std().item():.4f}')
        x = F.relu(self.bn128_1(self.conv128_1(x)))
        x = self.pool(F.relu(self.bn128_2(self.conv128_2(x))))
        # print(f'Block C3 mean: {x.mean().item():.4f}, std:{x.std().item():.4f}')
        x = F.relu(self.bn256_1(self.conv256_1(x)))
        x = self.pool(F.relu(self.bn256_2(self.conv256_2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.head(x)
        # print(f'Block FC mean: {out.mean().item():.4f}, std:{out.std().item():.4f}')
        # print('-------------')
        return out


class TwoHeadedCNN(CNNBase):
    def __init__(self, out_dim_hours, out_dim_minutes):
        super().__init__()
        self.head_hours = nn.Linear(128, out_dim_hours)
        self.head_minutes = nn.Linear(128, out_dim_minutes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn32(self.conv32(x))))
        x = self.pool(F.relu(self.bn64(self.conv64(x))))
        x = F.relu(self.bn128_1(self.conv128_1(x)))
        x = self.pool(F.relu(self.bn128_2(self.conv128_2(x))))
        x = F.relu(self.bn256_1(self.conv256_1(x)))
        x = self.pool(F.relu(self.bn256_2(self.conv256_2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out_hours = self.head_hours(x)
        out_minutes = self.head_minutes(x)
        return out_hours, out_minutes