# 손글씨 인식기 웹/데스크톱 분리 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 PyTorch + Tkinter 손글씨 인식기를 `desktop_version/`으로 옮기고, 외부 라이브러리 없이 순수 자바스크립트로 추론하는 `web_version/`을 새로 만든다.

**Architecture:** 파이썬이 학습과 가중치 내보내기를 맡고, 자바스크립트는 내보낸 Float32 바이너리를 `fetch`로 읽어 직접 구현한 conv/pool/linear 루프로 순전파한다. 파이썬이 만든 정답 데이터를 기준으로 자바스크립트 결과를 대조하며, 브라우저 검증 페이지가 그 대조를 수행한다.

**Tech Stack:** Python 3.11 / PyTorch 2.14 / NumPy 2.4 / Tkinter / 브라우저 내장 API (ES Modules, Canvas 2D, fetch) — 그 외 라이브러리 없음

**Spec:** `docs/superpowers/specs/2026-09-18-web-desktop-split-design.md`

## Global Constraints

- **기존 파이썬 파일 3개(`model.py`, `train.py`, `predict_gui.py`)와 `mnist_cnn.pt`는 한 글자도 수정하지 않는다.** 이동은 `git mv`로만 한다.
- **웹 버전은 외부 라이브러리를 일절 쓰지 않는다.** ONNX Runtime, TensorFlow.js, npm 의존성, 번들러, CDN 스크립트 모두 금지.
- **빌드 단계가 없어야 한다.** GitHub Pages가 저장소 파일을 그대로 서빙하면 동작해야 한다.
- 파이썬 실행 경로: `C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe`
- 파이썬 스크립트는 모두 `desktop_version/` 안에서 실행한다 (상대 경로 `./data`, `mnist_cnn.pt` 때문).
- 정규화 상수: mean `0.1307`, std `0.3081`. **자바스크립트 코드에 이 숫자를 적지 않는다.** `mnist_cnn.json`에서 읽는다.
- 텐서 순서(고정): `conv1.weight`, `conv1.bias`, `conv2.weight`, `conv2.bias`, `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias` — 총 421,642개, 1,686,568 바이트.
- 모든 코드 주석과 UI 텍스트는 한글로 쓴다 (기존 저장소 관행).
- 커밋 메시지 끝에 `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` 를 붙인다.

## 성공 기준 (전 태스크 완료 시점)

| 항목 | 기준 | 확인 위치 |
|---|---|---|
| 순전파 일치 | 확률 최대 절대차 ≤ 1e-4 | Task 4 |
| 전체 정확도 | 200장 중 ≥ 97% | Task 5 |
| 데스크톱 회귀 | 이동 후에도 동일 동작 | Task 1 |
| 웹 앱 동작 | 인식 + 상위 3개 후보 표시 | Task 6 |

## 파일 구조

| 파일 | 책임 | 태스크 |
|---|---|---|
| `desktop_version/model.py`, `train.py`, `predict_gui.py`, `mnist_cnn.pt` | 기존 코드 (이동만) | 1 |
| `desktop_version/CLAUDE.md` | 데스크톱 실행법 + 경로 주의 | 1 |
| `desktop_version/export_weights.py` | `.pt` → `.bin` + 매니페스트 | 2 |
| `desktop_version/make_test_data.py` | 파이썬 기준 전처리 + 정답 데이터 생성 | 3 |
| `web_version/js/nn.js` | conv2d / relu / maxPool2d / linear / softmax | 4 |
| `web_version/js/model.js` | 가중치 로더 + 순전파 조립 | 4 |
| `web_version/test.html` | 검증 페이지 (1단계 → 2단계) | 4, 5 |
| `web_version/js/preprocess.js` | 280x280 → 정규화된 28x28 | 5 |
| `web_version/js/draw.js` | 캔버스 그리기만 담당 | 6 |
| `web_version/js/app.js` | 세 모듈 연결 + 화면 갱신 | 6 |
| `web_version/index.html`, `css/style.css` | 앱 화면 | 6 |
| `CLAUDE.md`, `web_version/CLAUDE.md` | 루트/웹 문서 + 실측값 기록 | 7 |

---

### Task 1: 폴더 분리와 데스크톱 회귀 확인

**Files:**
- Move: `model.py`, `train.py`, `predict_gui.py`, `mnist_cnn.pt` → `desktop_version/`
- Move: `data/` → `desktop_version/data/` (깃 추적 대상 아님)
- Modify: `.gitignore`
- Create: `desktop_version/CLAUDE.md`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: `desktop_version/` 안의 기존 파이썬 모듈. 이후 모든 파이썬 스크립트는 이 폴더에서 실행되며 `from model import MnistCNN`으로 모델을 가져온다.

- [ ] **Step 1: 이동 전 기준선을 기록한다**

이동 후 같은 결과가 나오는지 비교할 기준이다.

```bash
cd D:/웹개발/Study01_MNIST; C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe -c "import torch; from model import MnistCNN; m=MnistCNN(); m.load_state_dict(torch.load('mnist_cnn.pt', map_location='cpu')); m.eval(); print('OK', sum(p.numel() for p in m.parameters()), tuple(m(torch.zeros(1,1,28,28)).shape))"
```

기대 출력: `OK 421642 (1, 10)`

- [ ] **Step 2: 폴더를 만들고 추적 중인 파일 4개를 `git mv`로 옮긴다**

```bash
cd D:/웹개발/Study01_MNIST; mkdir desktop_version; git mv model.py train.py predict_gui.py mnist_cnn.pt desktop_version/
```

- [ ] **Step 3: 추적되지 않는 `data/`를 옮긴다**

`data/`는 `.gitignore` 대상이라 `git mv`가 아니라 일반 이동이다.

```bash
cd D:/웹개발/Study01_MNIST; mv data desktop_version/data
```

- [ ] **Step 4: `.gitignore`를 갱신한다**

`data/`와 `__pycache__/` 패턴은 경로를 지정하지 않아 하위 폴더에서도 그대로 동작하므로 건드리지 않는다. 검증 픽스처 한 줄만 추가한다.

`.gitignore` 최종 내용:

```
__pycache__/
*.pyc
data/
web_version/model/test_data.json
```

- [ ] **Step 5: 이동 후 동일하게 동작하는지 확인한다 (회귀 확인 1/3)**

```bash
cd D:/웹개발/Study01_MNIST/desktop_version; C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe -c "import torch; from model import MnistCNN; m=MnistCNN(); m.load_state_dict(torch.load('mnist_cnn.pt', map_location='cpu')); m.eval(); print('OK', sum(p.numel() for p in m.parameters()), tuple(m(torch.zeros(1,1,28,28)).shape))"
```

기대 출력: Step 1과 **완전히 동일**한 `OK 421642 (1, 10)`

- [ ] **Step 6: `train.py`의 데이터 경로가 살아 있는지 확인한다 (회귀 확인 2/3)**

전체 학습은 돌리지 않고, `./data`의 MNIST를 찾아 로더가 준비되는지까지만 본다.

```bash
cd D:/웹개발/Study01_MNIST/desktop_version; C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe -c "import train; tr, te = train.get_data_loaders(); print('학습', len(tr.dataset), '테스트', len(te.dataset))"
```

기대 출력: `학습 60000 테스트 10000` — 다운로드가 다시 시작되면 `data/` 이동이 잘못된 것이다.

- [ ] **Step 7: 루트에서 실행하면 실패하는지 확인한다 (회귀 확인 3/3)**

상대 경로 제약이 실제로 존재함을 확인하는 단계다. **실패해야 정상이다.**

```bash
cd D:/웹개발/Study01_MNIST; C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe -c "import torch; torch.load('mnist_cnn.pt')"
```

기대: `FileNotFoundError` — 이 동작 때문에 CLAUDE.md에 실행 위치를 적는다.

- [ ] **Step 8: GUI가 실제로 뜨는지 눈으로 확인한다**

창이 뜨고, 숫자를 그린 뒤 "인식하기"를 눌러 결과가 나오면 통과. 확인 후 "종료"로 닫는다.

```bash
cd D:/웹개발/Study01_MNIST/desktop_version; C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe predict_gui.py
```

- [ ] **Step 9: `desktop_version/CLAUDE.md`를 작성한다**

````markdown
# CLAUDE.md — 데스크톱 버전

MNIST CNN 학습과 Tkinter 손글씨 인식 GUI. 모든 주석과 UI 텍스트는 한글이다.

## 반드시 이 폴더 안에서 실행할 것

`train.py`는 `"./data"`를, `train.py`와 `predict_gui.py`는 `"mnist_cnn.pt"`를
상대 경로로 참조한다. 이 경로들은 **의도적으로 수정하지 않았다.** 저장소
루트나 다른 위치에서 실행하면 파일을 찾지 못한다.

Python 3.11이 필요하다. 이 머신에서는 다음 경로에 있다
(`python`/`python3`는 Microsoft Store 스텁으로 연결될 수 있다):

```
C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe
```

## 명령

의존성 설치:
```
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install pillow numpy
```

학습 (MNIST를 `./data`에 내려받고 5에폭 학습 후 `mnist_cnn.pt` 저장):
```
python train.py
```

GUI 실행 (`mnist_cnn.pt`가 이미 있어야 함):
```
python predict_gui.py
```

웹 버전용 가중치 내보내기 (`../web_version/model/`에 씀):
```
python export_weights.py
```

웹 버전 검증용 정답 데이터 생성 (`../web_version/model/test_data.json`, 깃에 넣지 않음):
```
python make_test_data.py
```

테스트 스위트, 린터, 빌드 단계는 없다.

## 구조

- `model.py` — `MnistCNN`. 이 저장소의 유일한 모델 정의이며 웹 버전도 이
  구조를 그대로 재현한다. conv(1→32)→pool→conv(32→64)→pool→dropout→
  fc(3136→128)→dropout→fc(128→10), 입력은 28x28 단일 채널.
  **이 구조를 바꾸면** `train.py`로 재학습하고 `export_weights.py`를 다시
  실행해야 한다. 그러지 않으면 웹 버전이 shape 불일치로 실패한다.
- `train.py` — MNIST를 평균 `0.1307` / 표준편차 `0.3081`로 정규화해 학습하고
  `mnist_cnn.pt`에 `state_dict`를 저장한다.
- `predict_gui.py` — 280x280 캔버스에 그린 그림을 28x28로 단순 축소해 예측한다.
  경계상자 정렬이나 무게중심 보정을 하지 않는다. 웹 버전은 이 보정을 하므로
  같은 그림에 다른 답이 나올 수 있다.
- `export_weights.py` — `mnist_cnn.pt`를 웹용 Float32 바이너리와 매니페스트로
  내보낸다. **정규화 상수 `MEAN`/`STD`를 모듈 상수로 들고 있으며, 이 값은
  `train.py`의 `transforms.Normalize` 인자와 반드시 일치해야 한다.**
  자바스크립트는 이 상수를 코드에 적지 않고 매니페스트에서 읽는다.
- `make_test_data.py` — 웹 검증용 정답 데이터를 만든다. 파이썬 기준 전처리
  구현이 이 파일 안에 들어 있는데, 오직 자바스크립트가 맞출 기준을 만들기
  위한 것이며 `predict_gui.py`는 사용하지 않는다.
- `data/`, `__pycache__/` — 생성물. 손대지 않는다.
````

- [ ] **Step 10: 커밋**

```bash
cd D:/웹개발/Study01_MNIST; git add -A; git commit -m "Move desktop code into desktop_version/

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: 가중치 내보내기

**Files:**
- Create: `desktop_version/export_weights.py`
- Create (생성물, 커밋함): `web_version/model/mnist_cnn.bin`, `web_version/model/mnist_cnn.json`

**Interfaces:**
- Consumes: Task 1의 `desktop_version/model.py`, `mnist_cnn.pt`
- Produces:
  - `mnist_cnn.bin` — Float32 리틀엔디안, 텐서 8개를 고정 순서로 이어붙임. 총 1,686,568 바이트.
  - `mnist_cnn.json` — `{"tensors": [{"name": str, "shape": [int], "offset": int, "count": int}, ...], "normalize": {"mean": float, "std": float}, "totalBytes": int}`. `offset`은 바이트 단위이며 항상 4의 배수다 (JS `new Float32Array(buffer, offset, count)` 요구사항).

- [ ] **Step 1: `desktop_version/export_weights.py`를 작성한다**

내보낸 파일을 되읽어 원본과 대조하는 검사를 스크립트 안에 넣는다. 검사가 코드와 한 파일에 있으므로 "실행해서 결과를 본다"가 곧 테스트다.

```python
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
```

- [ ] **Step 2: 실행해서 자체 검사를 통과하는지 본다**

```bash
cd D:/웹개발/Study01_MNIST/desktop_version; C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe export_weights.py
```

기대 출력의 마지막 줄들:
```
합계: 421642개, 1686568 바이트
정규화 상수: mean=0.1307, std=0.3081
```

숫자가 다르면 `model.py` 구조가 스펙과 어긋난 것이므로 멈추고 확인한다.

- [ ] **Step 3: 파일 크기를 독립적으로 확인한다**

```bash
cd D:/웹개발/Study01_MNIST; C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe -c "import os,json; p='web_version/model/'; print('bin 바이트:', os.path.getsize(p+'mnist_cnn.bin')); m=json.load(open(p+'mnist_cnn.json',encoding='utf-8')); print('매니페스트 totalBytes:', m['totalBytes']); print('텐서 개수:', len(m['tensors'])); print('정규화:', m['normalize'])"
```

기대: `bin 바이트: 1686568`, `매니페스트 totalBytes: 1686568`, `텐서 개수: 8`, `정규화: {'mean': 0.1307, 'std': 0.3081}`

- [ ] **Step 4: 커밋**

가중치 산출물은 배포에 필요하므로 함께 커밋한다.

```bash
cd D:/웹개발/Study01_MNIST; git add desktop_version/export_weights.py web_version/model/mnist_cnn.bin web_version/model/mnist_cnn.json; git commit -m "Add weight exporter for the web version

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: 정답 데이터 생성 (구현보다 먼저)

**Files:**
- Create: `desktop_version/make_test_data.py`
- Create (생성물, 깃 제외): `web_version/model/test_data.json`

**Interfaces:**
- Consumes: Task 1의 `model.py` / `mnist_cnn.pt`, Task 2의 `mnist_cnn.json` (정규화 상수를 여기서 읽어 일치시킨다)
- Produces: `test_data.json` = `{"count": 200, "items": [{"pixels": [784개 0~255 정수], "label": int, "preprocessed": [784개 float], "probs": [10개 float]}, ...]}`
  - 자바스크립트는 `pixels`를 10배 최근접 확대해 280x280을 만들고, `preprocessed`/`probs`를 기준값으로 쓴다.

- [ ] **Step 1: `desktop_version/make_test_data.py`를 작성한다**

여기 담긴 전처리 구현이 자바스크립트가 맞춰야 할 **기준**이다. `predict_gui.py`는 이것을 쓰지 않는다.

```python
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
MANIFEST_PATH = os.path.join("..", "web_version", "model", "mnist_cnn.json")
OUT_PATH = os.path.join("..", "web_version", "model", "test_data.json")

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
```

- [ ] **Step 2: 실행한다**

```bash
cd D:/웹개발/Study01_MNIST/desktop_version; C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe make_test_data.py
```

기대 출력:
- `매니페스트에서 읽은 정규화 상수: mean=0.1307, std=0.3081`
- `파이썬 기준 정확도: ...` — **97% 이상이어야 한다.** 이 값이 낮으면 자바스크립트가 아니라 전처리 구현 자체가 잘못된 것이므로, 다음 태스크로 넘어가지 말고 `preprocess`를 고친다.
- 파일 크기 약 1.9MB

- [ ] **Step 3: 픽스처가 깃에서 제외되는지 확인한다**

```bash
cd D:/웹개발/Study01_MNIST; git status --short; git check-ignore -v web_version/model/test_data.json
```

기대: `git status`에 `test_data.json`이 나타나지 않고, `check-ignore`가 `.gitignore:4:web_version/model/test_data.json` 를 출력한다.

- [ ] **Step 4: 커밋 (스크립트만)**

```bash
cd D:/웹개발/Study01_MNIST; git add desktop_version/make_test_data.py; git commit -m "Add fixture generator with the reference preprocessing

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: 순전파 구현과 1단계 검증

**Files:**
- Create: `web_version/test.html`
- Create: `web_version/js/nn.js`
- Create: `web_version/js/model.js`

**Interfaces:**
- Consumes: Task 2의 `mnist_cnn.bin` / `mnist_cnn.json`, Task 3의 `test_data.json`
- Produces:
  - `nn.js`: `conv2d(input, inC, inH, inW, weight, bias, outC) -> Float32Array`, `relu(x) -> Float32Array`(제자리), `maxPool2d(input, c, h, w) -> Float32Array`, `linear(input, weight, bias, outFeatures) -> Float32Array`, `softmax(logits) -> Float32Array`
  - `model.js`: `loadModel(basePath) -> Promise<{tensors, normalize: {mean, std}, predict(Float32Array(784)) -> Float32Array(10)}>`
  - `normalize`는 Task 5의 `preprocess.js`가 인자로 받아 쓴다.

- [ ] **Step 1: 실패하는 검증 페이지를 먼저 만든다**

`web_version/test.html`:

```html
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>손글씨 인식기 — 검증</title>
<style>
  body { font-family: "맑은 고딕", sans-serif; margin: 24px; line-height: 1.6; }
  h1 { font-size: 20px; }
  h2 { font-size: 16px; margin-top: 28px; }
  table { border-collapse: collapse; margin-top: 8px; }
  th, td { border: 1px solid #ccc; padding: 4px 10px; text-align: left; }
  .pass { color: #0a7d23; font-weight: bold; }
  .fail { color: #c02020; font-weight: bold; }
  #오류 { color: #c02020; white-space: pre-wrap; }
</style>
</head>
<body>
<h1>손글씨 인식기 검증</h1>
<p>이 페이지는 개발 도구다. <code>make_test_data.py</code>를 먼저 실행해
<code>model/test_data.json</code>을 만들어야 동작한다.</p>

<h2>1단계 — 순전파 일치</h2>
<div id="1단계">실행 중…</div>

<h2>2단계 — 전처리와 전체 정확도</h2>
<div id="2단계">1단계 완료 후 실행된다.</div>

<div id="오류"></div>

<script type="module">
import { loadModel } from "./js/model.js";
</script>
</body>
</html>
```

이 시점에는 `js/model.js`가 없으므로 페이지가 오류를 낸다. 그게 이번 단계에서 보려는 실패다.

- [ ] **Step 2: 실패를 확인한다**

서버를 띄운다 (이 서버는 이후 태스크에서도 계속 쓴다):

```bash
cd D:/웹개발/Study01_MNIST/web_version; C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe -m http.server 8000
```

브라우저에서 `http://localhost:8000/test.html`을 열고 개발자 도구 콘솔을 본다.
기대: `js/model.js` 404 오류.

- [ ] **Step 3: `web_version/js/nn.js`를 작성한다**

```javascript
// 신경망 레이어 연산. 외부 라이브러리 없이 Float32Array 위에서 직접 계산한다.
// 텐서는 모두 (채널, 행, 열) 순서로 평탄화된 1차원 배열로 다룬다.

// 3x3 합성곱, stride 1, padding 1 고정. 출력 크기는 입력과 같다.
export function conv2d(input, inC, inH, inW, weight, bias, outC) {
  const out = new Float32Array(outC * inH * inW);

  for (let oc = 0; oc < outC; oc++) {
    for (let oy = 0; oy < inH; oy++) {
      for (let ox = 0; ox < inW; ox++) {
        let sum = bias[oc];

        for (let ic = 0; ic < inC; ic++) {
          for (let ky = 0; ky < 3; ky++) {
            const iy = oy + ky - 1;            // padding 1
            if (iy < 0 || iy >= inH) continue;

            for (let kx = 0; kx < 3; kx++) {
              const ix = ox + kx - 1;
              if (ix < 0 || ix >= inW) continue;

              sum += input[(ic * inH + iy) * inW + ix] *
                     weight[((oc * inC + ic) * 3 + ky) * 3 + kx];
            }
          }
        }

        out[(oc * inH + oy) * inW + ox] = sum;
      }
    }
  }

  return out;
}

// 음수를 0으로 만든다. 입력 배열을 그대로 고쳐 쓰고 같은 배열을 돌려준다.
export function relu(x) {
  for (let i = 0; i < x.length; i++) {
    if (x[i] < 0) x[i] = 0;
  }
  return x;
}

// 2x2 최대 풀링, stride 2 고정. 크기가 절반이 된다.
export function maxPool2d(input, c, h, w) {
  const outH = h >> 1;
  const outW = w >> 1;
  const out = new Float32Array(c * outH * outW);

  for (let ch = 0; ch < c; ch++) {
    const base = ch * h * w;

    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        const y = oy * 2;
        const x = ox * 2;

        let best = input[base + y * w + x];
        const 후보1 = input[base + y * w + x + 1];
        const 후보2 = input[base + (y + 1) * w + x];
        const 후보3 = input[base + (y + 1) * w + x + 1];

        if (후보1 > best) best = 후보1;
        if (후보2 > best) best = 후보2;
        if (후보3 > best) best = 후보3;

        out[(ch * outH + oy) * outW + ox] = best;
      }
    }
  }

  return out;
}

// 완전연결층. weight는 PyTorch Linear 규약대로 (출력, 입력) 순서다.
export function linear(input, weight, bias, outFeatures) {
  const inFeatures = input.length;
  const out = new Float32Array(outFeatures);

  for (let o = 0; o < outFeatures; o++) {
    let sum = bias[o];
    const row = o * inFeatures;

    for (let i = 0; i < inFeatures; i++) {
      sum += input[i] * weight[row + i];
    }

    out[o] = sum;
  }

  return out;
}

// 점수(logit)를 확률로 바꾼다. 최댓값을 빼서 지수 폭주를 막는다.
export function softmax(logits) {
  const out = new Float32Array(logits.length);

  let max = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }

  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    const v = Math.exp(logits[i] - max);
    out[i] = v;
    total += v;
  }

  for (let i = 0; i < out.length; i++) {
    out[i] /= total;
  }

  return out;
}
```

- [ ] **Step 4: `web_version/js/model.js`를 작성한다**

```javascript
// 내보낸 가중치를 불러와 model.py의 forward와 같은 순서로 순전파한다.
// 드롭아웃은 예측 시 아무 일도 하지 않으므로 구현하지 않는다.

import { conv2d, relu, maxPool2d, linear, softmax } from "./nn.js";

// model.py의 구조에서 오는 고정값
const 입력한변 = 28;
const CONV1_출력 = 32;
const CONV2_출력 = 64;
const FC1_출력 = 128;
const 분류개수 = 10;

// 정규화된 28x28 배열(784개)을 받아 10개 확률을 돌려준다.
export function forward(tensors, input) {
  let x = conv2d(input, 1, 28, 28,
                 tensors["conv1.weight"], tensors["conv1.bias"], CONV1_출력);
  relu(x);
  x = maxPool2d(x, CONV1_출력, 28, 28);          // 32 x 14 x 14

  x = conv2d(x, CONV1_출력, 14, 14,
             tensors["conv2.weight"], tensors["conv2.bias"], CONV2_출력);
  relu(x);
  x = maxPool2d(x, CONV2_출력, 14, 14);          // 64 x 7 x 7 = 3136

  x = linear(x, tensors["fc1.weight"], tensors["fc1.bias"], FC1_출력);
  relu(x);

  x = linear(x, tensors["fc2.weight"], tensors["fc2.bias"], 분류개수);
  return softmax(x);
}

// basePath 아래의 mnist_cnn.json / mnist_cnn.bin을 읽어 모델을 준비한다.
export async function loadModel(basePath) {
  const 매니페스트응답 = await fetch(`${basePath}/mnist_cnn.json`);
  if (!매니페스트응답.ok) {
    throw new Error(`매니페스트를 불러오지 못했습니다 (${매니페스트응답.status}).`);
  }
  const manifest = await 매니페스트응답.json();

  const 가중치응답 = await fetch(`${basePath}/mnist_cnn.bin`);
  if (!가중치응답.ok) {
    throw new Error(`가중치 파일을 불러오지 못했습니다 (${가중치응답.status}).`);
  }
  const buffer = await 가중치응답.arrayBuffer();

  if (buffer.byteLength !== manifest.totalBytes) {
    throw new Error(
      `가중치 크기가 매니페스트와 다릅니다: ${buffer.byteLength} vs ${manifest.totalBytes}`
    );
  }

  const tensors = {};
  for (const entry of manifest.tensors) {
    tensors[entry.name] = new Float32Array(buffer, entry.offset, entry.count);
  }

  return {
    tensors,
    normalize: manifest.normalize,   // { mean, std } — 코드에 숫자를 적지 않는다
    입력한변,
    predict(input) {
      if (input.length !== 입력한변 * 입력한변) {
        throw new Error(`입력 길이가 ${입력한변 * 입력한변}이 아닙니다: ${input.length}`);
      }
      return forward(tensors, input);
    },
  };
}
```

- [ ] **Step 5: `test.html`의 스크립트를 실제 검증으로 바꾼다**

`test.html` 아래쪽 `<script type="module">` 블록 전체를 다음으로 교체한다.

```javascript
import { loadModel } from "./js/model.js";

const 순전파허용오차 = 1e-4;

function 표만들기(행들) {
  const table = document.createElement("table");

  행들.forEach((행, 행번호) => {
    const tr = document.createElement("tr");
    for (const 칸 of 행) {
      const td = document.createElement(행번호 === 0 ? "th" : "td");
      td.innerHTML = 칸;
      tr.appendChild(td);
    }
    table.appendChild(tr);
  });

  return table;
}

function 판정(통과) {
  return 통과
    ? '<span class="pass">통과</span>'
    : '<span class="fail">실패</span>';
}

function 최대위치(배열) {
  let 위치 = 0;
  for (let i = 1; i < 배열.length; i++) {
    if (배열[i] > 배열[위치]) 위치 = i;
  }
  return 위치;
}

async function 정답데이터읽기() {
  const response = await fetch("./model/test_data.json");
  if (!response.ok) {
    throw new Error(
      "model/test_data.json 을 찾을 수 없습니다. " +
      "desktop_version 폴더에서 make_test_data.py 를 먼저 실행하세요."
    );
  }
  return response.json();
}

async function 일단계(모델, 데이터) {
  let 최대오차 = 0;
  let 예측일치 = 0;
  const 실패목록 = [];

  for (let i = 0; i < 데이터.items.length; i++) {
    const 항목 = 데이터.items[i];
    const 확률 = 모델.predict(Float32Array.from(항목.preprocessed));

    let 오차 = 0;
    for (let k = 0; k < 10; k++) {
      오차 = Math.max(오차, Math.abs(확률[k] - 항목.probs[k]));
    }
    최대오차 = Math.max(최대오차, 오차);

    const js예측 = 최대위치(확률);
    const py예측 = 최대위치(항목.probs);
    if (js예측 === py예측) 예측일치++;
    else 실패목록.push(`${i}번: JS ${js예측} vs 파이썬 ${py예측}`);
  }

  const 개수 = 데이터.items.length;
  const 오차통과 = 최대오차 <= 순전파허용오차;
  const 예측통과 = 예측일치 === 개수;

  const 영역 = document.getElementById("1단계");
  영역.innerHTML = "";
  영역.appendChild(표만들기([
    ["항목", "기준", "실측", "판정"],
    ["확률 최대 절대차", `≤ ${순전파허용오차}`, 최대오차.toExponential(2), 판정(오차통과)],
    ["예측 숫자 일치", `${개수}/${개수}`, `${예측일치}/${개수}`, 판정(예측통과)],
  ]));

  if (실패목록.length > 0) {
    const pre = document.createElement("pre");
    pre.className = "fail";
    pre.textContent = 실패목록.slice(0, 20).join("\n");
    영역.appendChild(pre);
  }

  return 오차통과 && 예측통과;
}

async function 실행() {
  try {
    const [모델, 데이터] = await Promise.all([
      loadModel("./model"),
      정답데이터읽기(),
    ]);
    await 일단계(모델, 데이터);
  } catch (error) {
    document.getElementById("오류").textContent = error.message;
    document.getElementById("1단계").innerHTML = 판정(false);
  }
}

실행();
```

- [ ] **Step 6: 1단계 검증을 통과하는지 확인한다**

서버가 떠 있는 상태에서 `http://localhost:8000/test.html`을 새로고침한다.

기대: 1단계 표의 두 줄이 모두 **통과**이고, 확률 최대 절대차가 `1e-4`보다 훨씬 작다 (`1e-7` 수준이 정상).

실패하면 순서대로 의심한다: `fc1.weight` 평탄화 순서 → conv 인덱싱 → 풀링 경계.

- [ ] **Step 7: 실측값을 기록해 둔다**

표에 나온 최대 절대차 값을 메모한다. Task 7에서 `web_version/CLAUDE.md`에 적는다.

- [ ] **Step 8: 커밋**

```bash
cd D:/웹개발/Study01_MNIST; git add web_version/js/nn.js web_version/js/model.js web_version/test.html; git commit -m "Add pure-JS forward pass and stage 1 verification page

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: 전처리 구현과 2단계 검증

**Files:**
- Create: `web_version/js/preprocess.js`
- Modify: `web_version/test.html` (2단계 추가)

**Interfaces:**
- Consumes: Task 4의 `loadModel`, Task 3의 `test_data.json`
- Produces: `preprocess.js`
  - `upscaleNearest(pixels, size, factor) -> Float32Array` — 정수 배율 최근접 확대
  - `preprocess(image, width, height, mean, std) -> Float32Array(784) | null` — 비어 있으면 `null`
  - Task 6의 `app.js`가 캔버스 픽셀에 대해 `preprocess`를 호출한다.

- [ ] **Step 1: 2단계 검증을 먼저 `test.html`에 추가한다**

`import` 줄 아래에 `preprocess.js` import를 추가하고, `실행()` 함수 위에 `이단계`를 넣은 뒤, `실행()` 안에서 `일단계` 다음에 `await 이단계(모델, 데이터);`를 호출하도록 고친다.

```javascript
import { upscaleNearest, preprocess } from "./js/preprocess.js";

const 정확도기준 = 0.97;
const 확대배율 = 10;

async function 이단계(모델, 데이터) {
  let 최대전처리오차 = 0;
  let 맞힌개수 = 0;
  const 실패목록 = [];

  for (let i = 0; i < 데이터.items.length; i++) {
    const 항목 = 데이터.items[i];

    const 큰그림 = upscaleNearest(항목.pixels, 28, 확대배율);
    const 처리됨 = preprocess(큰그림, 280, 280, 모델.normalize.mean, 모델.normalize.std);

    if (처리됨 === null) {
      실패목록.push(`${i}번: 전처리가 null을 돌려줌`);
      continue;
    }

    for (let k = 0; k < 784; k++) {
      최대전처리오차 = Math.max(최대전처리오차, Math.abs(처리됨[k] - 항목.preprocessed[k]));
    }

    const 확률 = 모델.predict(처리됨);
    const 예측 = 최대위치(확률);
    if (예측 === 항목.label) 맞힌개수++;
    else 실패목록.push(`${i}번: 예측 ${예측}, 정답 ${항목.label}`);
  }

  const 개수 = 데이터.items.length;
  const 정확도 = 맞힌개수 / 개수;
  const 정확도통과 = 정확도 >= 정확도기준;

  const 영역 = document.getElementById("2단계");
  영역.innerHTML = "";
  영역.appendChild(표만들기([
    ["항목", "기준", "실측", "판정"],
    ["전체 정확도", `≥ ${(정확도기준 * 100).toFixed(0)}%`,
     `${(정확도 * 100).toFixed(1)}% (${맞힌개수}/${개수})`, 판정(정확도통과)],
    ["전처리 최대 오차", "참고용 (기준 없음)", 최대전처리오차.toExponential(2), "—"],
  ]));

  const 설명 = document.createElement("p");
  설명.textContent =
    "전처리는 픽셀 단위 일치를 요구하지 않는다. 파이썬과 브라우저의 " +
    "리샘플링 구현이 다르기 때문이며, 판단 기준은 전체 정확도다.";
  영역.appendChild(설명);

  if (실패목록.length > 0) {
    const pre = document.createElement("pre");
    pre.textContent = 실패목록.slice(0, 20).join("\n");
    영역.appendChild(pre);
  }

  return 정확도통과;
}
```

- [ ] **Step 2: 실패를 확인한다**

`http://localhost:8000/test.html` 새로고침.
기대: `js/preprocess.js` 404로 페이지가 오류를 낸다.

- [ ] **Step 3: `web_version/js/preprocess.js`를 작성한다**

`make_test_data.py`의 `preprocess`와 같은 단계를 같은 순서로 수행한다.

```javascript
// 그려진 그림을 MNIST 규약에 맞춘 28x28 정규화 배열로 바꾼다.
// 이 파일에는 정규화 상수를 적지 않는다. mnist_cnn.json에서 읽어 인자로 받는다.

const 캔버스한변 = 28;      // 최종 출력 한 변
const 목표상자 = 20;        // 숫자를 담을 정사각형 한 변 (MNIST 원본 규약)

// 정수 배율 최근접 확대. 보간이 없어 파이썬 np.repeat과 결과가 같다.
export function upscaleNearest(pixels, size, factor) {
  const 새한변 = size * factor;
  const out = new Float32Array(새한변 * 새한변);

  for (let y = 0; y < 새한변; y++) {
    const sy = Math.floor(y / factor);
    for (let x = 0; x < 새한변; x++) {
      const sx = Math.floor(x / factor);
      out[y * 새한변 + x] = pixels[sy * size + sx];
    }
  }

  return out;
}

// 면적 평균 축소. 출력 픽셀이 덮는 원본 영역을 실수 구간으로 잡고
// 겹치는 넓이를 가중치로 평균낸다. make_test_data.py와 같은 식이다.
function 면적평균축소(src, srcW, srcH, outW, outH) {
  const out = new Float64Array(outW * outH);

  for (let oy = 0; oy < outH; oy++) {
    const y0 = (oy * srcH) / outH;
    const y1 = ((oy + 1) * srcH) / outH;

    for (let ox = 0; ox < outW; ox++) {
      const x0 = (ox * srcW) / outW;
      const x1 = ((ox + 1) * srcW) / outW;

      let 합 = 0;
      let 가중치합 = 0;

      for (let sy = Math.floor(y0); sy < Math.ceil(y1); sy++) {
        const wy = Math.min(y1, sy + 1) - Math.max(y0, sy);
        if (wy <= 0) continue;

        for (let sx = Math.floor(x0); sx < Math.ceil(x1); sx++) {
          const wx = Math.min(x1, sx + 1) - Math.max(x0, sx);
          if (wx <= 0) continue;

          const 넓이 = wy * wx;
          합 += src[sy * srcW + sx] * 넓이;
          가중치합 += 넓이;
        }
      }

      out[oy * outW + ox] = 가중치합 > 0 ? 합 / 가중치합 : 0;
    }
  }

  return out;
}

// image: 그레이스케일 값 배열 (검은 배경 0, 흰 획 255)
// 아무것도 그려지지 않았으면 null을 돌려준다.
export function preprocess(image, width, height, mean, std) {
  // 1) 경계상자 찾기
  let 위 = height, 아래 = -1, 왼쪽 = width, 오른쪽 = -1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (image[y * width + x] > 0) {
        if (y < 위) 위 = y;
        if (y > 아래) 아래 = y;
        if (x < 왼쪽) 왼쪽 = x;
        if (x > 오른쪽) 오른쪽 = x;
      }
    }
  }

  if (아래 < 0) return null;

  // 2) 잘라내기
  const 자른너비 = 오른쪽 - 왼쪽 + 1;
  const 자른높이 = 아래 - 위 + 1;
  const 자름 = new Float64Array(자른너비 * 자른높이);

  for (let y = 0; y < 자른높이; y++) {
    for (let x = 0; x < 자른너비; x++) {
      자름[y * 자른너비 + x] = image[(위 + y) * width + (왼쪽 + x)];
    }
  }

  // 3) 긴 변이 20px가 되도록 비율 유지 축소
  const 배율 = 목표상자 / Math.max(자른높이, 자른너비);
  const 새높이 = Math.max(1, Math.round(자른높이 * 배율));
  const 새너비 = Math.max(1, Math.round(자른너비 * 배율));
  const 작은그림 = 면적평균축소(자름, 자른너비, 자른높이, 새너비, 새높이);

  // 4) 경계상자 기준으로 가운데 놓기
  const 판 = new Float64Array(캔버스한변 * 캔버스한변);
  const 시작y = Math.floor((캔버스한변 - 새높이) / 2);
  const 시작x = Math.floor((캔버스한변 - 새너비) / 2);

  for (let y = 0; y < 새높이; y++) {
    for (let x = 0; x < 새너비; x++) {
      판[(시작y + y) * 캔버스한변 + (시작x + x)] = 작은그림[y * 새너비 + x];
    }
  }

  // 5) 무게중심을 중앙으로 옮기기
  let 총합 = 0, y가중 = 0, x가중 = 0;
  for (let y = 0; y < 캔버스한변; y++) {
    for (let x = 0; x < 캔버스한변; x++) {
      const v = 판[y * 캔버스한변 + x];
      총합 += v;
      y가중 += v * y;
      x가중 += v * x;
    }
  }

  let 최종 = 판;
  if (총합 > 0) {
    const 중심 = (캔버스한변 - 1) / 2;
    const 이동y = Math.round(중심 - y가중 / 총합);
    const 이동x = Math.round(중심 - x가중 / 총합);

    최종 = new Float64Array(캔버스한변 * 캔버스한변);
    for (let y = 0; y < 캔버스한변; y++) {
      const ty = y + 이동y;
      if (ty < 0 || ty >= 캔버스한변) continue;

      for (let x = 0; x < 캔버스한변; x++) {
        const tx = x + 이동x;
        if (tx < 0 || tx >= 캔버스한변) continue;

        최종[ty * 캔버스한변 + tx] = 판[y * 캔버스한변 + x];
      }
    }
  }

  // 6) 정규화
  const out = new Float32Array(캔버스한변 * 캔버스한변);
  for (let i = 0; i < out.length; i++) {
    out[i] = (최종[i] / 255 - mean) / std;
  }

  return out;
}
```

**주의:** `Math.round`는 파이썬 `round`와 .5 처리가 다르다(파이썬은 짝수 반올림). 무게중심 이동량이 정확히 정수 경계에 걸릴 일은 드물고, 걸리더라도 1픽셀 차이라 정확도에 영향이 없다. 2단계 정확도가 기준을 넘으면 그대로 둔다.

- [ ] **Step 4: 2단계 검증을 통과하는지 확인한다**

`http://localhost:8000/test.html` 새로고침.

기대: 전체 정확도 **97% 이상**. 1단계도 계속 통과해야 한다.

기준 미달이면 스펙의 판단 기준에 따라 Lanczos 축소를 직접 구현해 다시 측정한다. 그 전에 먼저 확인할 것: `make_test_data.py`가 출력한 파이썬 기준 정확도와 비교해 큰 차이가 나는지. 차이가 크면 자바스크립트 전처리 구현이 어긋난 것이고, 둘 다 낮으면 전처리 방식 자체의 문제다.

- [ ] **Step 5: 실측값을 기록해 둔다**

전체 정확도와 전처리 최대 오차를 메모한다. Task 7에서 사용한다.

- [ ] **Step 6: 커밋**

```bash
cd D:/웹개발/Study01_MNIST; git add web_version/js/preprocess.js web_version/test.html; git commit -m "Add MNIST-style preprocessing and stage 2 verification

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: 웹 앱 화면

**Files:**
- Create: `web_version/js/draw.js`
- Create: `web_version/js/app.js`
- Create: `web_version/index.html`
- Create: `web_version/css/style.css`

**Interfaces:**
- Consumes: Task 4의 `loadModel`, Task 5의 `preprocess`
- Produces: 브라우저에서 동작하는 앱.
  - `draw.js`: `createBoard(canvasElement) -> {getPixels(): Float32Array, clear(): void}`

- [ ] **Step 1: `web_version/js/draw.js`를 작성한다**

그리기만 담당한다. 추론을 알지 못한다.

```javascript
// 캔버스 그림판. 마우스와 터치로 선을 긋고, 그레이스케일 픽셀 배열을 넘겨준다.
// 추론이나 전처리에 대해서는 아무것도 모른다.

const 펜굵기 = 18;   // predict_gui.py의 PEN_WIDTH와 같은 값

export function createBoard(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  let 그리는중 = false;

  function 바탕칠하기() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function 좌표(event) {
    // 캔버스가 CSS로 늘어나 있을 수 있으므로 실제 픽셀 좌표로 환산한다
    const 영역 = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - 영역.left) * (canvas.width / 영역.width),
      y: (event.clientY - 영역.top) * (canvas.height / 영역.height),
    };
  }

  바탕칠하기();
  ctx.strokeStyle = "white";
  ctx.lineWidth = 펜굵기;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  canvas.addEventListener("pointerdown", (event) => {
    그리는중 = true;
    canvas.setPointerCapture(event.pointerId);

    const p = 좌표(event);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(p.x, p.y);   // 점 하나만 찍어도 자국이 남도록
    ctx.stroke();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!그리는중) return;

    const p = 좌표(event);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  });

  function 끝내기() {
    if (!그리는중) return;
    그리는중 = false;
    ctx.closePath();
  }

  canvas.addEventListener("pointerup", 끝내기);
  canvas.addEventListener("pointercancel", 끝내기);

  // 터치로 그릴 때 화면이 스크롤되지 않게 한다
  canvas.style.touchAction = "none";

  return {
    getPixels() {
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      const out = new Float32Array(canvas.width * canvas.height);

      for (let i = 0; i < out.length; i++) {
        out[i] = data[i * 4];   // 흰 획이므로 R 채널만 봐도 충분하다
      }

      return out;
    },

    clear() {
      바탕칠하기();
    },
  };
}
```

- [ ] **Step 2: `web_version/index.html`을 작성한다**

```html
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>손글씨 숫자 인식기</title>
<link rel="stylesheet" href="./css/style.css">
</head>
<body>
<main>
  <h1>손글씨 숫자 인식기</h1>
  <p class="안내">아래 칸에 숫자 하나를 크게 그린 뒤 "인식하기"를 누르세요.</p>

  <canvas id="그림판" width="280" height="280"></canvas>

  <div class="버튼줄">
    <button id="인식">인식하기</button>
    <button id="지우기">지우기</button>
  </div>

  <p id="결과">모델을 불러오는 중…</p>

  <section id="확률표" hidden>
    <h2>숫자별 확률</h2>
    <div id="막대들"></div>
  </section>
</main>

<script type="module" src="./js/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: `web_version/css/style.css`를 작성한다**

```css
:root {
  --강조: #1a73e8;
  --흐림: #9aa0a6;
  --글자: #202124;
}

body {
  font-family: "맑은 고딕", "Malgun Gothic", sans-serif;
  color: var(--글자);
  margin: 0;
  padding: 24px 16px 48px;
  display: flex;
  justify-content: center;
}

main { width: 100%; max-width: 420px; }

h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 20px 0 8px; }

.안내 { color: var(--흐림); font-size: 14px; margin: 0 0 16px; }

#그림판 {
  width: 100%;
  max-width: 280px;
  aspect-ratio: 1 / 1;
  background: black;
  border-radius: 8px;
  cursor: crosshair;
  display: block;
}

.버튼줄 { display: flex; gap: 8px; margin: 12px 0; }

button {
  font: inherit;
  padding: 10px 18px;
  border: 1px solid #dadce0;
  border-radius: 6px;
  background: white;
  cursor: pointer;
}

button:hover:not(:disabled) { background: #f1f3f4; }
button:disabled { opacity: 0.5; cursor: default; }

#결과 { font-size: 20px; font-weight: bold; min-height: 28px; margin: 8px 0; }

.막대줄 {
  display: grid;
  grid-template-columns: 20px 1fr 56px;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
  font-size: 13px;
  color: var(--흐림);
}

.막대틀 { background: #f1f3f4; border-radius: 3px; height: 10px; }
.막대 { background: var(--흐림); border-radius: 3px; height: 100%; }

.막대줄.상위 { color: var(--글자); font-weight: bold; }
.막대줄.상위 .막대 { background: var(--강조); }
```

- [ ] **Step 4: `web_version/js/app.js`를 작성한다**

```javascript
// 그림판, 전처리, 모델을 연결하고 결과를 화면에 그린다.

import { createBoard } from "./draw.js";
import { preprocess } from "./preprocess.js";
import { loadModel } from "./model.js";

const 강조개수 = 3;   // 상위 몇 개를 강조할지

const 캔버스 = document.getElementById("그림판");
const 결과 = document.getElementById("결과");
const 확률표 = document.getElementById("확률표");
const 막대들 = document.getElementById("막대들");
const 인식버튼 = document.getElementById("인식");
const 지우기버튼 = document.getElementById("지우기");

const 그림판 = createBoard(캔버스);
let 모델 = null;

인식버튼.disabled = true;

function 막대그리기(확률) {
  // 확률이 높은 순으로 상위 몇 개를 미리 골라둔다
  const 순위 = [...확률.keys()]
    .sort((a, b) => 확률[b] - 확률[a])
    .slice(0, 강조개수);

  막대들.innerHTML = "";

  for (let 숫자 = 0; 숫자 < 10; 숫자++) {
    const 줄 = document.createElement("div");
    줄.className = 순위.includes(숫자) ? "막대줄 상위" : "막대줄";

    const 이름 = document.createElement("span");
    이름.textContent = 숫자;

    const 틀 = document.createElement("div");
    틀.className = "막대틀";

    const 막대 = document.createElement("div");
    막대.className = "막대";
    막대.style.width = `${(확률[숫자] * 100).toFixed(2)}%`;
    틀.appendChild(막대);

    const 값 = document.createElement("span");
    값.textContent = `${(확률[숫자] * 100).toFixed(1)}%`;

    줄.append(이름, 틀, 값);
    막대들.appendChild(줄);
  }

  확률표.hidden = false;
}

function 인식하기() {
  if (!모델) return;

  const 픽셀 = 그림판.getPixels();
  const 입력 = preprocess(
    픽셀, 캔버스.width, 캔버스.height,
    모델.normalize.mean, 모델.normalize.std
  );

  if (입력 === null) {
    결과.textContent = "숫자를 그려주세요";
    확률표.hidden = true;
    return;
  }

  const 확률 = 모델.predict(입력);

  let 예측 = 0;
  for (let i = 1; i < 확률.length; i++) {
    if (확률[i] > 확률[예측]) 예측 = i;
  }

  결과.textContent = `예측 결과: ${예측}  (확신도: ${(확률[예측] * 100).toFixed(1)}%)`;
  막대그리기(확률);
}

function 지우기() {
  그림판.clear();
  결과.textContent = "숫자를 그려주세요";
  확률표.hidden = true;
}

인식버튼.addEventListener("click", 인식하기);
지우기버튼.addEventListener("click", 지우기);

loadModel("./model")
  .then((준비된모델) => {
    모델 = 준비된모델;
    인식버튼.disabled = false;
    결과.textContent = "숫자를 그려주세요";
  })
  .catch((error) => {
    결과.textContent = `모델을 불러오지 못했습니다: ${error.message}`;
  });
```

- [ ] **Step 5: 브라우저에서 직접 확인한다**

서버가 떠 있는 상태에서 `http://localhost:8000/` 을 연다. 다음을 모두 확인한다.

1. "숫자를 그려주세요"가 표시되고 "인식하기" 버튼이 활성화된다
2. 숫자를 그리고 "인식하기"를 누르면 예측 숫자와 확신도가 나온다
3. 0~9 막대가 모두 보이고 **상위 3개가 파란색으로 강조**된다
4. "지우기"를 누르면 캔버스와 확률표가 비워진다
5. 아무것도 안 그린 채 "인식하기"를 누르면 "숫자를 그려주세요"가 뜬다
6. 0부터 9까지 하나씩 그려보고 대부분 맞히는지 본다

- [ ] **Step 6: 콘솔에 오류가 없는지 확인한다**

개발자 도구 콘솔을 열어 오류와 경고가 없는지 본다. 특히 `Float32Array` 오프셋 정렬 오류나 모듈 로딩 실패가 없어야 한다.

- [ ] **Step 7: 커밋**

```bash
cd D:/웹개발/Study01_MNIST; git add web_version/js/draw.js web_version/js/app.js web_version/index.html web_version/css/style.css; git commit -m "Add web app UI with drawing board and probability bars

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: 문서 정리와 실측값 기록

**Files:**
- Modify: `CLAUDE.md` (루트 — 전면 교체)
- Create: `web_version/CLAUDE.md`

**Interfaces:**
- Consumes: Task 4~6에서 메모해 둔 실측값
- Produces: 없음 (마지막 태스크)

- [ ] **Step 1: 실측값을 다시 확인한다**

기억에 의존하지 않는다. `http://localhost:8000/test.html`을 다시 열어 표의 값을 그대로 옮겨 적는다.

- [ ] **Step 2: 루트 `CLAUDE.md`를 다시 쓴다**

````markdown
# CLAUDE.md

이 파일은 Claude Code가 이 저장소에서 작업할 때 참고할 지침이다.

## 개요

MNIST 손글씨 숫자 인식기. 같은 CNN을 두 가지 방식으로 제공한다.

- **[desktop_version/](desktop_version/)** — PyTorch로 학습하고 Tkinter GUI로 인식한다.
  자세한 내용은 [desktop_version/CLAUDE.md](desktop_version/CLAUDE.md).
- **[web_version/](web_version/)** — 외부 라이브러리 없이 순수 자바스크립트로 추론한다.
  GitHub Pages에 그대로 올라간다. 자세한 내용은 [web_version/CLAUDE.md](web_version/CLAUDE.md).

모든 코드 주석과 UI 텍스트는 한글로 쓴다.

## 두 버전이 함께 지켜야 하는 것

- **모델 구조** — conv(1→32) → pool → conv(32→64) → pool → fc(3136→128) → fc(128→10),
  입력은 28x28 단일 채널. 정의는 `desktop_version/model.py` 한 곳뿐이고,
  웹은 `web_version/js/model.js`에서 이를 재현한다.
- **정규화 상수** — 평균 `0.1307`, 표준편차 `0.3081`. 자바스크립트는 이 값을
  코드에 적지 않고 `web_version/model/mnist_cnn.json`에서 읽는다.
- **입력 형식** — 28x28 단일 채널, 검은 배경에 흰 글씨.

**모델 구조를 바꾸면** 반드시 이 순서를 지킨다:

```
cd desktop_version
python train.py            # 재학습 -> mnist_cnn.pt
python export_weights.py   # 재내보내기 -> web_version/model/
python make_test_data.py   # 정답 데이터 재생성
```

그러지 않으면 웹 버전이 shape 불일치로 실패한다.

## 두 버전의 답이 다를 수 있다

의도된 차이다. `predict_gui.py`는 280x280 캔버스를 그대로 28x28로 축소할 뿐이고,
웹 버전은 MNIST 원본 규약대로 경계상자를 자르고 20x20으로 맞춘 뒤 무게중심을
가운데로 옮긴다. **보통 웹 버전이 더 정확하다.** 기존 파이썬 코드를 수정하지
않기로 했기 때문에 이 차이는 그대로 둔다.

## 문서

- 설계: [docs/superpowers/specs/2026-09-18-web-desktop-split-design.md](docs/superpowers/specs/2026-09-18-web-desktop-split-design.md)
- 구현 계획: [docs/superpowers/plans/2026-09-18-web-desktop-split.md](docs/superpowers/plans/2026-09-18-web-desktop-split.md)

## 생성물

`desktop_version/data/`, `__pycache__/`, `web_version/model/test_data.json`은
생성물이라 깃에 없다. 손으로 고치지 않는다.

반면 `web_version/model/mnist_cnn.bin`과 `mnist_cnn.json`은 배포된 앱이
동작하려면 반드시 있어야 하므로 커밋한다. 빌드 단계가 없기 때문이다.

테스트 스위트, 린터, 빌드 도구는 없다. 검증은 `web_version/test.html`로 한다.
````

- [ ] **Step 3: `web_version/CLAUDE.md`를 작성한다**

`<...>` 자리에 Step 1에서 확인한 실제 값을 넣는다.

````markdown
# CLAUDE.md — 웹 버전

브라우저에서 손글씨 숫자를 인식한다. 학습된 CNN의 순전파를 자바스크립트로
직접 구현했다.

## 절대 규칙

**외부 라이브러리를 쓰지 않는다.** ONNX Runtime, TensorFlow.js, npm 패키지,
번들러, CDN 스크립트 모두 금지다. 브라우저 내장 기능(ES Modules, Canvas 2D,
fetch)만 쓴다. 빌드 단계도 없다 — GitHub Pages가 이 폴더를 그대로 서빙하면
동작해야 한다.

전체 연산량은 약 420만 MAC이라 프레임워크 없이도 즉시 끝난다.

## 실행

`.bin`을 `fetch`로 읽기 때문에 `index.html`을 파일로 직접 열면(`file://`)
동작하지 않는다. 서버를 띄운다.

```
cd web_version
python -m http.server 8000
```

- `http://localhost:8000/` — 앱
- `http://localhost:8000/test.html` — 검증 페이지

검증 페이지는 `model/test_data.json`이 필요하다. 이 파일은 깃에 없으므로
먼저 만들어야 한다.

```
cd desktop_version
python make_test_data.py
```

## 배포

GitHub Pages는 브랜치 배포에서 루트 또는 `/docs`만 고를 수 있다. 루트 배포를
쓰며, 주소는 다음과 같다.

```
https://<사용자>.github.io/Study01_MNIST/web_version/
```

모든 경로가 상대 경로라 설정 파일이 필요 없다. `test.html`은 픽스처가 깃에
없어 배포본에서 동작하지 않는다. 검증 페이지는 개발 도구이지 배포 대상이 아니다.

## 파일별 책임

- `js/nn.js` — 레이어 연산만. `conv2d`(3x3, stride 1, padding 1 고정),
  `relu`(제자리), `maxPool2d`(2x2, stride 2 고정), `linear`, `softmax`.
  모두 순수 함수이고 모델을 알지 못한다. 드롭아웃은 예측 시 항등이라 없다.
- `js/model.js` — `loadModel(basePath)`이 매니페스트와 `.bin`을 읽어
  `Float32Array` 뷰를 자르고, `predict(입력784개)`가 `model.py`의 `forward`와
  같은 순서로 순전파한다.
- `js/preprocess.js` — 280x280 그림을 정규화된 28x28로 바꾼다.
  **정규화 상수를 이 파일에 적지 않는다.** `mnist_cnn.json`에서 읽어 인자로 받는다.
- `js/draw.js` — 캔버스 그리기만. 추론을 알지 못한다.
- `js/app.js` — 위 셋을 연결하고 화면을 갱신한다.

## 가중치 형식

`desktop_version/export_weights.py`가 만든다. 직접 고치지 않는다.

- `model/mnist_cnn.bin` — Float32 리틀엔디안, 텐서 8개를 고정 순서로 이어붙임.
  421,642개 값, 1,686,568 바이트.
- `model/mnist_cnn.json` — 텐서별 이름/shape/바이트 오프셋/개수, 그리고 정규화
  상수 `mean`/`std`. 오프셋은 항상 4의 배수다
  (`new Float32Array(buffer, offset, count)` 요구사항).

## 전처리

MNIST 원본 데이터의 생성 방식을 따른다.

1. 그려진 픽셀의 경계상자를 자른다 (비어 있으면 안내 메시지)
2. 비율을 유지한 채 긴 변이 20px가 되도록 면적 평균 축소
3. 잉크의 무게중심이 28x28 중앙에 오도록 정수 평행이동
4. `/255` 후 `(x - mean) / std`

축소는 PIL의 LANCZOS 대신 면적 평균을 직접 구현했다. 브라우저에서 LANCZOS를
그대로 재현할 수 없기 때문이며, 정확도가 기준을 넘어 Lanczos는 구현하지 않았다.

`desktop_version/predict_gui.py`는 이 보정을 하지 않는다. 같은 그림에 두 버전이
다른 답을 낼 수 있다.

## 측정 결과

`test.html`로 MNIST 테스트 200장에 대해 측정한 값이다.

| 항목 | 기준 | 실측 |
|---|---|---|
| 순전파 일치 (확률 최대 절대차) | ≤ 1e-4 | <1단계 표의 값> |
| 전체 정확도 | ≥ 97% | <2단계 표의 값> |

모델 구조나 가중치를 바꾸면 다시 측정해 이 표를 갱신한다.
````

- [ ] **Step 4: 문서의 링크와 명령이 맞는지 확인한다**

```bash
cd D:/웹개발/Study01_MNIST; ls desktop_version web_version web_version/js web_version/model docs/superpowers/specs docs/superpowers/plans
```

CLAUDE.md들이 언급한 파일이 모두 실제로 존재하는지 대조한다.

- [ ] **Step 5: 최종 상태를 확인한다**

```bash
cd D:/웹개발/Study01_MNIST; git status --short; git ls-files
```

기대: `test_data.json`, `data/`, `__pycache__/`가 추적 목록에 없다.

- [ ] **Step 6: 커밋**

```bash
cd D:/웹개발/Study01_MNIST; git add -A; git commit -m "Document both versions and record measured results

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## 완료 확인

모든 태스크가 끝나면 네 항목을 한 번에 다시 돌려 본다. 통과를 주장하기 전에
실제 출력을 확인한다.

1. `test.html` 1단계 — 확률 최대 절대차 ≤ 1e-4
2. `test.html` 2단계 — 전체 정확도 ≥ 97%
3. `desktop_version/`에서 `predict_gui.py` 실행 — 창이 뜨고 인식됨
4. `http://localhost:8000/` — 그린 숫자를 인식하고 상위 3개 후보 표시
