# -*- coding: utf-8 -*-
"""학습된 가중치(mnist_cnn.pt)를 웹 버전이 읽을 수 있는 형식으로 내보내는 스크립트

Float32 리틀엔디안 바이너리 하나와, 텐서 위치를 설명하는 매니페스트 JSON을 만든다.
자바스크립트는 매니페스트를 보고 하나의 ArrayBuffer 위에 Float32Array 뷰를 자른다.
"""

import json
import os

import numpy as np
import torch

WEIGHT_PATH = "mnist_cnn.pt"
OUT_DIR = os.path.join("..", "web_version", "model")
BIN_NAME = "mnist_cnn.bin"
JSON_NAME = "mnist_cnn.json"

# train.py의 transforms.Normalize 인자와 반드시 일치해야 한다.
# 자바스크립트는 이 값을 코드에 적지 않고 매니페스트에서 읽어 쓴다.
MEAN = 0.1307
STD = 0.3081

# 내보내는 순서. 자바스크립트는 이름으로 찾으므로 순서 자체가 규약은 아니지만,
# 바이트 오프셋을 정하는 기준이 되므로 고정해 둔다.
TENSOR_ORDER = [
    "conv1.weight", "conv1.bias",
    "conv2.weight", "conv2.bias",
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
]


def load_state_dict():
    """저장된 가중치를 불러온다"""
    if not os.path.exists(WEIGHT_PATH):
        raise SystemExit(
            f"'{WEIGHT_PATH}' 파일이 없습니다. 먼저 이 폴더에서 train.py를 실행하세요."
        )
    return torch.load(WEIGHT_PATH, map_location="cpu")


def build_payload(state_dict):
    """텐서를 순서대로 평탄화해 바이트열과 매니페스트 항목을 만든다"""
    chunks = []
    entries = []
    offset = 0

    for name in TENSOR_ORDER:
        if name not in state_dict:
            raise SystemExit(f"가중치에 '{name}' 텐서가 없습니다. model.py 구조를 확인하세요.")

        array = state_dict[name].detach().cpu().numpy().astype("<f4")
        raw = array.tobytes()

        entries.append({
            "name": name,
            "shape": list(array.shape),
            "offset": offset,
            "count": int(array.size),
        })
        chunks.append(raw)
        offset += len(raw)

    return b"".join(chunks), entries, offset


def verify(blob, entries, state_dict):
    """내보낸 바이트열을 되읽어 원본 텐서와 정확히 일치하는지 확인한다"""
    for entry in entries:
        restored = np.frombuffer(
            blob, dtype="<f4", count=entry["count"], offset=entry["offset"]
        )
        original = state_dict[entry["name"]].detach().cpu().numpy().astype("<f4").ravel()

        if not np.array_equal(restored, original):
            raise SystemExit(f"'{entry['name']}' 텐서가 내보내기 후 달라졌습니다.")

        if entry["offset"] % 4 != 0:
            raise SystemExit(
                f"'{entry['name']}'의 오프셋 {entry['offset']}이 4의 배수가 아닙니다. "
                "자바스크립트에서 Float32Array 뷰를 만들 수 없습니다."
            )


def main():
    state_dict = load_state_dict()
    blob, entries, total_bytes = build_payload(state_dict)
    verify(blob, entries, state_dict)

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, BIN_NAME), "wb") as f:
        f.write(blob)

    manifest = {
        "tensors": entries,
        "normalize": {"mean": MEAN, "std": STD},
        "totalBytes": total_bytes,
    }
    with open(os.path.join(OUT_DIR, JSON_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("텐서별 내보내기 결과:")
    for entry in entries:
        print(f"  {entry['name']:<14} {str(entry['shape']):<18} "
              f"{entry['count']:>7}개  오프셋 {entry['offset']:>9}")

    total_count = sum(e["count"] for e in entries)
    print(f"합계: {total_count}개, {total_bytes} 바이트")
    print(f"정규화 상수: mean={MEAN}, std={STD}")
    print(f"'{os.path.join(OUT_DIR, BIN_NAME)}' 와 "
          f"'{os.path.join(OUT_DIR, JSON_NAME)}' 을 만들었습니다.")


if __name__ == "__main__":
    main()
