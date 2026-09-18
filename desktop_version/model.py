# -*- coding: utf-8 -*-
"""MNIST 손글씨 숫자 인식을 위한 CNN 모델 정의"""

import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """MNIST(28x28 흑백 손글씨 숫자) 분류를 위한 간단한 CNN"""

    def __init__(self):
        super().__init__()
        # 입력: 1채널(흑백) 28x28 이미지
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.pool = nn.MaxPool2d(2, 2)                             # 크기를 절반으로 축소

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # conv1 -> pool -> conv2 -> pool 을 거치면 28x28 -> 14x14 -> 7x7 이 됨
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 0~9, 총 10개 클래스

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)                 # 28x28 -> 14x14

        x = F.relu(self.conv2(x))
        x = self.pool(x)                 # 14x14 -> 7x7
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)        # 완전연결층에 넣기 위해 1차원으로 펼침
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)                  # 각 숫자(0~9)에 대한 점수(logit) 출력

        return x
