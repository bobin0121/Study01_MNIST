# 손글씨 인식기 웹/데스크톱 분리 설계

작성일: 2026-09-18

## 목적

현재 단일 폴더에 있는 PyTorch + Tkinter 손글씨 숫자 인식기를 두 개의
독립적인 버전으로 나눈다.

- `desktop_version/` — 기존 파이썬 코드. 학습과 Tkinter GUI 담당.
- `web_version/` — 새로 작성. 외부 라이브러리 없이 순수 자바스크립트로
  추론하며, GitHub Pages에 정적으로 배포 가능해야 한다.

## 제약

1. **기존 파이썬 파일은 수정하지 않는다.** `model.py`, `train.py`,
   `predict_gui.py`는 한 글자도 고치지 않고 `desktop_version/`으로 옮긴다.
   이동은 `git mv`로 처리해 커밋 이력을 보존한다.
2. **웹 버전은 외부 라이브러리를 일절 쓰지 않는다.** ONNX Runtime, TensorFlow.js,
   번들러, npm 의존성 모두 사용 금지. 브라우저 내장 기능만 사용한다.
3. **빌드 단계가 없어야 한다.** GitHub Pages가 저장소의 파일을 그대로 서빙하면
   동작해야 한다. 따라서 가중치 산출물도 저장소에 커밋한다.

## 최종 폴더 구조

```
Study01_MNIST/
├── CLAUDE.md                      # 루트: 두 버전 개요 + 공유 불변식
├── .gitignore
├── docs/superpowers/specs/
├── desktop_version/
│   ├── CLAUDE.md
│   ├── model.py                   # 수정 없이 이동
│   ├── train.py                   # 수정 없이 이동
│   ├── predict_gui.py             # 수정 없이 이동
│   ├── mnist_cnn.pt               # 수정 없이 이동
│   ├── data/                      # 이동 (gitignore 대상)
│   ├── export_weights.py          # 신규
│   └── make_test_data.py          # 신규
└── web_version/
    ├── CLAUDE.md
    ├── index.html
    ├── test.html
    ├── css/style.css
    ├── js/nn.js                   # 순수 JS 레이어 연산
    ├── js/model.js                # MnistCNN 순전파 + 가중치 로더
    ├── js/preprocess.js           # 캔버스 → 28x28 정규화
    ├── js/draw.js                 # 캔버스 그리기 (그림판)
    ├── js/app.js                  # UI 연결
    └── model/
        ├── mnist_cnn.bin          # Float32 가중치 (~1.61MB)
        ├── mnist_cnn.json         # 텐서 매니페스트
        └── test_data.json         # 검증용 픽스처 (gitignore 대상)
```

### 이동 후 실행 경로 주의

`train.py`는 `"./data"`를, `train.py`와 `predict_gui.py`는
`"mnist_cnn.pt"`를 상대 경로로 참조한다. 코드를 고치지 않으므로 두
스크립트는 반드시 `desktop_version/` 안에서 실행해야 한다. 이 제약은
`desktop_version/CLAUDE.md`에 명시한다.

## 가중치 내보내기

`export_weights.py`가 `mnist_cnn.pt`를 읽어 두 파일을 만든다.

**`mnist_cnn.bin`** — 아래 순서로 텐서를 평탄화해 이어붙인 Float32
리틀엔디안 바이너리. 총 421,642개 값 = 1,686,568 바이트.

| 순서 | 텐서 | shape | 개수 |
|---|---|---|---|
| 1 | conv1.weight | (32, 1, 3, 3) | 288 |
| 2 | conv1.bias | (32,) | 32 |
| 3 | conv2.weight | (64, 32, 3, 3) | 18,432 |
| 4 | conv2.bias | (64,) | 64 |
| 5 | fc1.weight | (128, 3136) | 401,408 |
| 6 | fc1.bias | (128,) | 128 |
| 7 | fc2.weight | (10, 128) | 1,280 |
| 8 | fc2.bias | (10,) | 10 |

**`mnist_cnn.json`** — 두 가지를 담는 매니페스트.

1. 각 텐서의 이름, shape, 바이트 오프셋, 원소 개수. 자바스크립트는
   이것을 보고 하나의 `ArrayBuffer` 위에 `Float32Array` 뷰를 자른다.
2. 정규화 상수 `mean`(0.1307)과 `std`(0.3081).

정규화 상수를 자바스크립트에 직접 적지 않고 이 파일에서 읽는 이유는
**값의 출처를 하나로 유지하기 위해서**다. 상수가 두 언어에 각각 적혀
있으면 한쪽만 바뀔 때 조용히 틀린 답을 내게 된다.

다만 이 저장소에는 상수를 모아둔 모듈이 없다. `0.1307`과 `0.3081`은
`train.py`의 `transforms.Normalize`와 `predict_gui.py` 안에 각각
하드코딩되어 있고, 두 파일은 수정하지 않는다. 따라서
`export_weights.py`가 상수를 모듈 상수로 보유하고 매니페스트에 기록한다.
자바스크립트 쪽에는 값이 전혀 적히지 않으며, 이 상수가 `train.py`와
일치해야 한다는 점을 `desktop_version/CLAUDE.md`에 명시한다.

`fc1.weight`는 PyTorch `Linear` 규약대로 `(out, in)` 순서이며, 입력
3136개는 conv2 출력을 `(채널, 행, 열)` 순으로 평탄화한 것에 대응한다
(`x.view(x.size(0), -1)`와 동일한 순서).

## 순수 자바스크립트 추론

연산량은 약 420만 MAC(conv2가 361만으로 대부분)이라 단순 루프로 충분히
빠르다. `Float32Array` 위에서 직접 계산한다.

**`nn.js`** — 순수 함수 모음. 각 함수는 입력 `Float32Array`와 shape을
받아 새 `Float32Array`를 반환한다.

- `conv2d(input, inShape, weight, bias, outChannels)` — 3x3, stride 1, padding 1 고정
- `relu(x)` — 제자리 연산
- `maxPool2d(input, inShape)` — 2x2, stride 2 고정
- `linear(input, weight, bias, outFeatures)`
- `softmax(logits)`

드롭아웃은 추론 시 항등 함수이므로 구현하지 않는다.

**`model.js`** — `loadModel(basePath)`이 매니페스트와 `.bin`을 `fetch`로
받아 텐서 맵을 만들고, `predict(input28x28)`이 순전파를 수행해 10개
확률을 반환한다. 순서는 `model.py`의 `forward`와 정확히 일치한다:

```
conv1 → relu → pool → conv2 → relu → pool → flatten → fc1 → relu → fc2 → softmax
```

## 웹 버전 전처리

MNIST 원본 데이터의 생성 방식을 그대로 따른다.

1. 캔버스에서 픽셀 값을 읽는다 (검은 배경에 흰 획).
2. 값이 0보다 큰 픽셀의 경계상자를 구한다. 비어 있으면 안내 메시지를
   띄우고 추론하지 않는다.
3. 종횡비를 유지한 채 긴 변이 20px가 되도록 **면적 평균**으로 축소한다.
4. 28x28 빈 배열에, 잉크의 **무게중심**이 중앙(13.5, 13.5)에 오도록
   정수 평행이동해 배치한다.
5. `/255` 후 `(x - mean) / std`로 정규화한다. 두 상수는 코드에 적지 않고
   `mnist_cnn.json`에서 읽어 쓴다.

### 데스크톱과의 차이 (의도된 것)

`predict_gui.py`는 280x280을 그대로 28x28로 축소할 뿐 경계상자 정렬이나
무게중심 보정을 하지 않는다. 기존 파이썬 코드를 수정하지 않기로 했으므로
이 차이는 유지된다. **같은 그림에 두 버전이 다른 답을 낼 수 있으며,
보통 웹 버전이 더 정확하다.** 이 사실은 루트 CLAUDE.md에 기록한다.

## 화면

- **`index.html`** — 280x280 검은 캔버스, "인식하기" / "지우기" 버튼,
  예측 숫자와 확신도, 그리고 0~9 전체 확률 막대. 막대는 0부터 9까지
  순서대로 놓되 **확률 상위 3개를 강조**해, 후보가 바로 읽히게 한다.
  마우스와 터치를 모두 지원한다. UI 텍스트는 한글.
- **`draw.js`** — 캔버스 그리기만 담당. 포인터 이벤트로 선을 잇고,
  `getPixels()`로 그레이스케일 배열을, `clear()`로 초기화를 제공한다.
  추론 로직을 알지 못한다.
- **`app.js`** — `draw.js`, `preprocess.js`, `model.js`를 연결하고 결과를
  화면에 그린다.

## 검증

이 저장소에는 테스트 프레임워크가 없다. 프레임워크를 도입하지 않고,
파이썬 결과를 기준으로 삼아 자바스크립트가 같은 답을 내는지 대조한다.

**구현보다 정답 데이터를 먼저 만든다.** 검사 수단을 먼저 두고 그것을
통과하도록 구현하는 순서다(test-driven-development).

### 정답 데이터: `make_test_data.py`

MNIST 테스트셋에서 200장을 골라 `web_version/model/test_data.json`을
만든다. 각 항목은 다음을 담는다.

| 필드 | 내용 |
|---|---|
| `pixels` | 원본 28x28 픽셀 (0~255 정수) |
| `label` | 정답 라벨 |
| `preprocessed` | 파이썬 전처리를 거친 28x28 정규화 결과 |
| `probs` | PyTorch가 계산한 10개 확률 |

파일 크기는 약 1.9MB이며 **깃에 넣지 않는다.** 생성물이므로
`.gitignore`에 `web_version/model/test_data.json`을 등록한다. 검증을
돌리려면 먼저 `make_test_data.py`를 실행해야 하며(torch와 MNIST 필요),
GitHub Pages에 올린 `test.html`은 이 파일을 못 찾아 동작하지 않는다.
검증 페이지는 개발 도구이지 배포 대상이 아니다. 실행법은
`web_version/CLAUDE.md`에 적는다.

반면 가중치 `mnist_cnn.bin`과 `mnist_cnn.json`은 배포된 앱이 동작하려면
반드시 있어야 하므로 그대로 커밋한다.

**280x280 그림을 저장하지 않는 이유**: 200장×78,400픽셀은 JSON으로
약 62MB가 되어 브라우저에서 다루기 무겁다. 대신 양쪽 모두 28x28을
**정확히 10배 최근접 확대**해 280x280을 만든다. 배율이 정수라 보간이
없고, 파이썬과 자바스크립트가 픽셀 단위로 동일한 그림을 얻는다.

**파이썬 전처리 기준 구현은 `make_test_data.py` 안에 둔다.** 이
저장소에는 `preprocess.py`가 없고 `predict_gui.py`는 수정하지 않으므로,
이 코드는 오직 자바스크립트가 맞춤 기준을 만들기 위해 존재한다. 데스크톱
GUI는 이것을 사용하지 않는다.

### 1단계: 순전파만 검증

`test.html`이 `preprocessed` 필드를 그대로 JS 순전파에 넣어 200장 전부에
대해 다음을 확인한다.

- 예측 숫자가 PyTorch 예측과 일치하는가
- 각 확률의 최대 절대 오차가 1e-4 이하인가

전처리를 일부러 비켜가는 것은 순서 때문이다. 순전파는 28x28 입력만
있으면 단독으로 검증할 수 있지만, 전처리는 순전파까지 이어 붙여야 결과를
판단할 수 있다. 따라서 순전파를 먼저 만들어 통과시킨 뒤 전처리를 붙인다.

### 2단계: 전처리 대조 + 전체 정확도

전처리가 완성되면 `test.html`에 두 번째 구역을 둔다. `pixels`를 10배
확대한 280x280을 JS 전처리에 통과시키고 두 가지를 확인한다.

- JS 전처리 결과가 `preprocessed`와 얼마나 가까운가 (픽셀별 최대 오차 표시)
- 전처리 + 순전파 전체를 거친 200장의 전체 정확도

전처리는 픽셀 단위 일치를 요구하지 않는다. 파이썬과 브라우저의
리샘플링 구현이 다르기 때문이다. 판단 기준은 **전체 정확도**다.

**축소 방식의 판단 기준**: 면적 평균 축소로 97% 이상이면 그대로 둘 것.
미달하면 그때 Lanczos를 직접 구현해 다시 측정한다. 성능을 미리 가정해서
복잡한 리샘플링을 먼저 쓰지 않는다. 측정 결과는 `web_version/CLAUDE.md`에
기록한다.

하나라도 실패하면 페이지가 붉게 표시하고 실패 항목을 나열한다.

### 가중치 내보내기 자체 검사

`export_weights.py`는 내보낸 직후 `.bin`을 되읽어 원본 텐서와 정확히
일치하는지 확인하고, 텐서별 개수와 총 바이트 수를 출력한다.

## GitHub Pages 배포

Pages의 브랜치 배포는 루트 또는 `/docs`만 선택할 수 있으므로 **루트
배포**를 사용한다. 저장소 전체가 서빙되며 웹 버전 주소는 다음과 같다.

```
https://<사용자>.github.io/Study01_MNIST/web_version/
```

모든 경로가 상대 경로라 별도 설정이나 워크플로 파일이 필요 없다.

**로컬 확인**: `.bin`을 `fetch`로 받기 때문에 `index.html`을 파일로 직접
열면(`file://`) CORS 정책 때문에 동작하지 않는다. 반드시 간단한 HTTP
서버를 띄워야 한다.

```
cd web_version && python -m http.server 8000
```

## 공유 불변식

두 버전이 반드시 같이 유지해야 하는 것:

- **모델 아키텍처** — conv(1→32) → pool → conv(32→64) → pool → fc(3136→128) → fc(128→10)
- **정규화 상수** — 평균 0.1307, 표준편차 0.3081. 웹은 이 값을 직접 갖지 않고
  `mnist_cnn.json`을 통해 받는다.
- **입력 형식** — 28x28 단일 채널, 검은 배경에 흰 글씨

`model.py`의 구조를 바꾸면 `train.py`로 재학습한 뒤 `export_weights.py`를
다시 실행해야 한다. 그러지 않으면 웹 버전은 shape이 맞지 않아 실패한다.

## CLAUDE.md 3종

- **루트** — 두 버전 개요, 공유 불변식, 재학습/재내보내기 규칙, 두 버전의
  전처리 차이
- **`desktop_version/`** — 기존 내용 + 이동 후 실행 경로 주의 +
  `export_weights.py` / `make_test_data.py` 사용법
- **`web_version/`** — 외부 라이브러리 금지 원칙, 가중치 형식, 파일별 책임,
  로컬 서버 실행법, Pages 배포법

## 성공 기준

구현이 끝난 것으로 보려면 네 항목이 모두 통과해야 한다. 각 항목은
주장이 아니라 실제로 돌려본 결과로 확인하고, 실측값을
`web_version/CLAUDE.md`에 기록한다.

| 항목 | 기준 | 확인 방법 |
|---|---|---|
| 순전파 일치 | 같은 28x28 입력에 대한 확률의 최대 절대차 **1e-4 이하** | `test.html` 1단계, 200장 |
| 전체 정확도 | 200장 중 **97% 이상** | `test.html` 2단계 |
| 데스크톱 회귀 | 이동 후에도 기존과 동일하게 동작 | 아래 참조 |
| 웹 앱 동작 | 그린 숫자를 인식하고 상위 3개 후보를 보여줌 | 브라우저에서 직접 그려보기 |

### 데스크톱 회귀 확인

참조한 설명서는 macOS의 `.app` 더블클릭을 포함하지만, 이 저장소는
Windows이고 앱 번들이 없다. 대응되는 확인은 다음 둘이며, 모두
`desktop_version/` 안에서 실행한다.

- `predict_gui.py` — `mnist_cnn.pt`를 읽고 창이 뜨며, 그린 숫자를 인식하는가
- `train.py` — `./data`의 MNIST를 찾아 데이터 로더가 준비되는가
  (전체 학습은 돌리지 않고 경로 해석까지만 확인)

파이썬 코드는 수정하지 않으므로, 이 확인은 상대 경로가 깨지지
않았는지만 본다.

## 범위 밖

- 기존 파이썬 코드의 동작 변경
- 웹에서의 학습 기능
- 모델 양자화나 파일 크기 최적화
- 실시간(획을 그리는 도중) 인식
