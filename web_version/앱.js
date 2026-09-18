// 그림판, 전처리, 모델을 연결하고 결과를 화면에 그린다.

import { createBoard } from "./그림판.js";
import { preprocess } from "./전처리.js";
import { loadModel } from "./모델.js";

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

loadModel(".")
  .then((준비된모델) => {
    모델 = 준비된모델;
    인식버튼.disabled = false;
    결과.textContent = "숫자를 그려주세요";
  })
  .catch((error) => {
    결과.textContent = `모델을 불러오지 못했습니다: ${error.message}`;
  });
