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
| 순전파 일치 (확률 최대 절대차) | ≤ 1e-4 | 2.41e-7 |
| 전체 정확도 | ≥ 97% | 98.0% (196/200) |

모델 구조나 가중치를 바꾸면 다시 측정해 이 표를 갱신한다.
