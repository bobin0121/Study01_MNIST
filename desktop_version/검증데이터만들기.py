# -*- coding: utf-8 -*-
"""웹 버전 검증에 쓸 정답 데이터를 만드는 스크립트

MNIST 테스트 이미지 200장에 대해 다음을 기록한다.
  pixels       원본 28x28 (0~255 정수)
  label        정답 라벨
  preprocessed 파이썬 기준 전처리를 거친 28x28 정규화 결과
  probs        PyTorch가 계산한 10개 확률

자바스크립트는 pixels를 10배 확대해 280x280을 만든 뒤 자기 전처리를 돌리고,
그 결과와 순전파 결과를 preprocessed / probs와 대조한다.

이 파일 안의 전처리 구현은 오직 그 기준을 만들기 위한 것이며,
predict_gui.py는 사용하지 않는다.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from model import MnistCNN

WEIGHT_PATH = "mnist_cnn.pt"
MANIFEST_PATH = os.path.join("..", "web_version", "가중치정보.json")
OUT_PATH = os.path.join("..", "web_version", "검증데이터.json")

SAMPLE_COUNT = 200
UPSCALE = 10          # 28 * 10 = 280, 앱 캔버스와 같은 크기
TARGET_BOX = 20       # 숫자를 담을 정사각형 한 변 (MNIST 원본 규약)
CANVAS = 28


def upscale_nearest(img28, factor):
    """28x28을 정수 배율로 최근접 확대한다 (보간 없음 -> 자바스크립트와 결과 동일)"""
    return np.repeat(np.repeat(img28, factor, axis=0), factor, axis=1)


def area_average_resize(src, out_h, out_w):
    """면적 평균으로 축소한다

    출력 픽셀 하나가 덮는 원본 영역을 실수 구간으로 잡고, 겹치는 넓이를
    가중치로 평균낸다. 자바스크립트에서 같은 식으로 구현하면 같은 값이 나온다.
    """
    src_h, src_w = src.shape
    out = np.zeros((out_h, out_w), dtype=np.float64)

    for oy in range(out_h):
        y0 = oy * src_h / out_h
        y1 = (oy + 1) * src_h / out_h

        for ox in range(out_w):
            x0 = ox * src_w / out_w
            x1 = (ox + 1) * src_w / out_w

            total = 0.0
            weight = 0.0

            for sy in range(int(np.floor(y0)), int(np.ceil(y1))):
                wy = min(y1, sy + 1) - max(y0, sy)
                if wy <= 0:
                    continue

                for sx in range(int(np.floor(x0)), int(np.ceil(x1))):
                    wx = min(x1, sx + 1) - max(x0, sx)
                    if wx <= 0:
                        continue

                    area = wy * wx
                    total += src[sy, sx] * area
                    weight += area

            out[oy, ox] = total / weight if weight > 0 else 0.0

    return out


def preprocess(img, mean, std):
    """280x280 그림을 MNIST 규약에 맞춘 28x28 정규화 배열로 바꾼다

    1) 그려진 픽셀의 경계상자를 자른다
    2) 종횡비를 유지한 채 긴 변이 20px가 되도록 면적 평균 축소
    3) 잉크의 무게중심이 28x28 중앙에 오도록 정수 평행이동
    4) /255 후 (x - mean) / std

    아무것도 그려지지 않았으면 None을 돌려준다.
    """
    ys, xs = np.nonzero(img > 0)
    if len(xs) == 0:
        return None

    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float64)
    crop_h, crop_w = crop.shape

    scale = TARGET_BOX / max(crop_h, crop_w)
    new_h = max(1, int(round(crop_h * scale)))
    new_w = max(1, int(round(crop_w * scale)))
    small = area_average_resize(crop, new_h, new_w)

    # 먼저 경계상자 기준으로 가운데 놓는다
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.float64)
    top = (CANVAS - new_h) // 2
    left = (CANVAS - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small

    # 그 다음 무게중심을 중앙으로 옮긴다
    total = canvas.sum()
    if total > 0:
        rows = np.arange(CANVAS).reshape(-1, 1)
        cols = np.arange(CANVAS).reshape(1, -1)
        center_y = float((canvas * rows).sum() / total)
        center_x = float((canvas * cols).sum() / total)

        shift_y = int(round((CANVAS - 1) / 2.0 - center_y))
        shift_x = int(round((CANVAS - 1) / 2.0 - center_x))

        shifted = np.zeros_like(canvas)
        for y in range(CANVAS):
            ty = y + shift_y
            if ty < 0 or ty >= CANVAS:
                continue
            for x in range(CANVAS):
                tx = x + shift_x
                if tx < 0 or tx >= CANVAS:
                    continue
                shifted[ty, tx] = canvas[y, x]
        canvas = shifted

    return (canvas / 255.0 - mean) / std


def main():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    mean = manifest["normalize"]["mean"]
    std = manifest["normalize"]["std"]
    print(f"매니페스트에서 읽은 정규화 상수: mean={mean}, std={std}")

    model = MnistCNN()
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
    model.eval()

    dataset = datasets.MNIST(root="./data", train=False, download=True)

    items = []
    for index in range(SAMPLE_COUNT):
        image, label = dataset[index]
        pixels = np.array(image, dtype=np.uint8)

        big = upscale_nearest(pixels, UPSCALE)
        processed = preprocess(big, mean, std)
        if processed is None:
            raise SystemExit(f"{index}번 이미지가 비어 있습니다.")

        # 자바스크립트는 JSON에 적힌(반올림된) 값을 받으므로, 확률도 같은 값에서
        # 계산해야 1단계 비교가 완전히 같은 입력에 대한 비교가 된다.
        rounded = [round(v, 6) for v in processed.ravel().tolist()]

        tensor = torch.tensor(rounded, dtype=torch.float32).view(1, 1, CANVAS, CANVAS)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0]

        items.append({
            "pixels": pixels.ravel().tolist(),
            "label": int(label),
            "preprocessed": rounded,
            "probs": [round(v, 8) for v in probs.tolist()],
        })

        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{SAMPLE_COUNT}장 처리")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"count": len(items), "items": items}, f)

    correct = sum(1 for it in items if int(np.argmax(it["probs"])) == it["label"])
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"파이썬 기준 정확도: {correct}/{len(items)} ({100.0 * correct / len(items):.1f}%)")
    print(f"'{OUT_PATH}' 생성 완료 ({size_mb:.2f}MB)")


if __name__ == "__main__":
    main()
