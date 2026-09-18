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
