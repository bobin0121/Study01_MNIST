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
    // 이미 한 손가락(포인터)으로 그리는 중이면 새 포인터는 무시한다.
    // 그러지 않으면 beginPath가 다시 호출되어 두 획이 이어져 보인다.
    if (그리는중) return;

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
