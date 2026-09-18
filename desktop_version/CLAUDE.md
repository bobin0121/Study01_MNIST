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

## 정규화 상수가 세 곳에 흩어져 있다

`0.1307` / `0.3081`은 공유되는 것이 아니라 **복사되어** 있다.

| 위치 | 형태 |
|---|---|
| `train.py` | `transforms.Normalize((0.1307,), (0.3081,))` |
| `predict_gui.py` | `(tensor - 0.1307) / 0.3081` 직접 연산 |
| `export_weights.py` | 모듈 상수 `MEAN` / `STD` |

앞의 둘은 기존 코드라 **의도적으로 수정하지 않았다.** 셋째는 새로
추가한 것이고, 이 값이 매니페스트를 통해 자바스크립트로 전달된다.

정규화 값을 바꾸려면 **세 곳을 모두** 고치고 재학습해야 한다. 한쪽만
고쳐도 오류 없이 돌아가며, 정확도만 조용히 떨어진다.

## 구조

- `model.py` — `MnistCNN`. 이 저장소의 유일한 모델 정의이며 웹 버전도 이
  구조를 그대로 재현한다. conv(1→32)→pool→conv(32→64)→pool→dropout→
  fc(3136→128)→dropout→fc(128→10), 입력은 28x28 단일 채널.
  **이 구조를 바꾸면** `train.py`로 재학습하고 `export_weights.py`를 다시
  실행해야 한다. 그러지 않으면 웹 버전이 shape 불일치로 실패한다.
- `train.py` — `torchvision.datasets.MNIST`로 데이터를 받아 평균 `0.1307` /
  표준편차 `0.3081`로 정규화해 학습하고, 에폭마다 테스트 정확도를 평가한
  뒤 `mnist_cnn.pt`에 `state_dict`를 저장한다. 하이퍼파라미터는 파일 맨 위의
  모듈 상수다 — `BATCH_SIZE`(64), `EPOCHS`(5), `LEARNING_RATE`(0.001),
  `WEIGHT_PATH`. 학습 설정을 바꿀 때는 이 네 줄만 고치면 된다.
- `predict_gui.py` — Tkinter 앱(`DigitRecognizerApp`). 280x280 검은 캔버스에
  그린다.

  **화면과 또 하나의 이미지에 동시에 그린다.** 마우스 드래그 한 번에
  `self.canvas.create_line`(눈에 보이는 쪽)과 `self.draw.line`(메모리 속 PIL
  이미지)을 나란히 호출한다. Tkinter 캔버스에서 픽셀을 다시 읽어올 수가
  없기 때문이다. **모델에 실제로 들어가는 것은 PIL 이미지(`self.image`) 쪽**이므로,
  그리기를 고칠 때 한쪽만 고치면 화면과 인식 결과가 조용히 어긋난다.
  지우기도 둘 다 지워야 한다(`canvas.delete` + `draw.rectangle`).

  펜 굵기는 `PEN_WIDTH = 18`로 고정이다. 웹 버전의 `js/draw.js`도 같은 18을
  쓴다 — 한쪽을 바꾸면 두 버전의 획 굵기가 달라진다.

  예측할 때는 PIL 이미지를 28x28로 **단순 축소**(`Image.LANCZOS`)할 뿐, 경계상자 정렬이나
  무게중심 보정을 하지 않는다. 웹 버전은 이 보정을 하므로 같은 그림에 다른
  답이 나올 수 있다.

  가중치는 `load_state_dict`로 읽으므로 `model.py` 구조가 바뀐 채 재학습하지
  않으면 shape 불일치로 실패한다.
- `export_weights.py` — `mnist_cnn.pt`를 웹용 Float32 바이너리와 매니페스트로
  내보낸다. **정규화 상수 `MEAN`/`STD`를 모듈 상수로 들고 있으며, 이 값은
  `train.py`의 `transforms.Normalize` 인자와 반드시 일치해야 한다.**
  자바스크립트는 이 상수를 코드에 적지 않고 매니페스트에서 읽는다.
- `make_test_data.py` — 웹 검증용 정답 데이터를 만든다. 파이썬 기준 전처리
  구현이 이 파일 안에 들어 있는데, 오직 자바스크립트가 맞출 기준을 만들기
  위한 것이며 `predict_gui.py`는 사용하지 않는다.
- `data/`, `__pycache__/` — 생성물. 손대지 않는다.
