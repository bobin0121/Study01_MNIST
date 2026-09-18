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
