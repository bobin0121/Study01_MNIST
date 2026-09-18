# -*- coding: utf-8 -*-
"""마우스로 숫자를 직접 그려서 학습된 CNN 모델로 인식하는 GUI 프로그램"""

import tkinter as tk
from tkinter import messagebox

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from model import MnistCNN

WEIGHT_PATH = "mnist_cnn.pt"

# 그림판 캔버스 크기 (실제로는 이 이미지를 28x28로 축소해서 모델에 입력)
CANVAS_SIZE = 280
PEN_WIDTH = 18


class DigitRecognizerApp:
    """숫자를 그리고 인식 결과를 보여주는 tkinter 앱"""

    def __init__(self, root, model, device):
        self.root = root
        self.model = model
        self.device = device

        self.root.title("손글씨 숫자 인식기")

        # 실제 그림이 그려질 캔버스 (화면에 보이는 부분)
        self.canvas = tk.Canvas(
            root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross"
        )
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        # 캔버스와 동일한 내용을 담는 PIL 이미지 (모델 입력용으로 사용)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # 마우스 이벤트 연결
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_position)

        self.last_x, self.last_y = None, None

        # 결과를 보여줄 라벨
        self.result_label = tk.Label(root, text="숫자를 그려주세요", font=("맑은 고딕", 20))
        self.result_label.grid(row=1, column=0, columnspan=3, pady=5)

        # 버튼: 인식하기 / 지우기
        predict_button = tk.Button(root, text="인식하기", command=self.predict)
        predict_button.grid(row=2, column=0, padx=5, pady=10)

        clear_button = tk.Button(root, text="지우기", command=self.clear_canvas)
        clear_button.grid(row=2, column=1, padx=5, pady=10)

        quit_button = tk.Button(root, text="종료", command=root.quit)
        quit_button.grid(row=2, column=2, padx=5, pady=10)

    def paint(self, event):
        """마우스를 드래그하는 동안 선을 그림"""
        x, y = event.x, event.y

        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=PEN_WIDTH, fill="white",
                capstyle=tk.ROUND, smooth=True,
            )
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255, width=PEN_WIDTH,
            )

        self.last_x, self.last_y = x, y

    def reset_last_position(self, event):
        """마우스 버튼을 뗐을 때 좌표 초기화 (다음 획이 이어지지 않도록)"""
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        """캔버스와 이미지를 모두 지움"""
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result_label.config(text="숫자를 그려주세요")

    def preprocess_image(self):
        """그린 이미지를 MNIST 입력 형식(1x1x28x28, 정규화)에 맞게 변환"""
        # 280x280 -> 28x28로 축소 (MNIST 이미지 크기에 맞춤)
        small_image = self.image.resize((28, 28), Image.LANCZOS)

        tensor = torch.tensor(list(small_image.getdata()), dtype=torch.float32)
        tensor = tensor.view(1, 1, 28, 28) / 255.0

        # 학습 때 사용한 것과 동일한 정규화 값 적용
        tensor = (tensor - 0.1307) / 0.3081

        return tensor.to(self.device)

    def predict(self):
        """현재 그려진 숫자를 모델로 예측하고 결과를 화면에 표시"""
        input_tensor = self.preprocess_image()

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        digit = predicted.item()
        percent = confidence.item() * 100
        self.result_label.config(text=f"예측 결과: {digit}  (확신도: {percent:.1f}%)")


def load_model():
    """저장된 가중치 파일(mnist_cnn.pt)을 불러와 모델을 준비"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MnistCNN().to(device)

    try:
        state_dict = torch.load(WEIGHT_PATH, map_location=device)
    except FileNotFoundError:
        messagebox.showerror(
            "오류",
            f"'{WEIGHT_PATH}' 파일을 찾을 수 없습니다.\n먼저 train.py를 실행해서 모델을 학습시켜 주세요.",
        )
        raise SystemExit(1)

    model.load_state_dict(state_dict)
    model.eval()

    return model, device


def main():
    model, device = load_model()

    root = tk.Tk()
    DigitRecognizerApp(root, model, device)
    root.mainloop()


if __name__ == "__main__":
    main()
