# -*- coding: utf-8 -*-
"""MNIST 데이터셋으로 CNN 모델을 학습시키고 가중치를 mnist_cnn.pt로 저장하는 스크립트"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MnistCNN

# 학습에 사용할 하이퍼파라미터
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
WEIGHT_PATH = "mnist_cnn.pt"


def get_data_loaders():
    """MNIST 학습/테스트 데이터를 내려받아 DataLoader로 반환"""
    # MNIST 데이터셋의 평균/표준편차로 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """한 에폭(epoch) 동안 모델을 학습"""
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 200 == 0:
            print(f"[에폭 {epoch}] 배치 {batch_idx}/{len(train_loader)} - 손실: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"[에폭 {epoch}] 평균 손실: {avg_loss:.4f}")


def evaluate(model, device, test_loader):
    """테스트 데이터셋으로 모델의 정확도를 평가"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"테스트 정확도: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    train_loader, test_loader = get_data_loaders()

    model = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, device, train_loader, optimizer, criterion, epoch)
        evaluate(model, device, test_loader)

    # 학습된 가중치 저장
    torch.save(model.state_dict(), WEIGHT_PATH)
    print(f"학습된 가중치를 '{WEIGHT_PATH}' 파일로 저장했습니다.")


if __name__ == "__main__":
    main()
