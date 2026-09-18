// 그려진 그림을 MNIST 규약에 맞춘 28x28 정규화 배열로 바꾼다.
// 이 파일에는 정규화 상수를 적지 않는다. 가중치정보.json에서 읽어 인자로 받는다.

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
// 겹치는 넓이를 가중치로 평균낸다. 검증데이터만들기.py와 같은 식이다.
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
  // Math.round는 .5를 항상 올림한다. 검증데이터만들기.py의 파이썬 round는
  // 은행반올림(banker's rounding)이라 다르게 반올림될 수 있지만, 그 차이는
  // 최대 1px이고 정확도 기준을 넘기므로 의도적으로 받아들인다.
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
    // 여기도 Math.round와 파이썬 round(은행반올림)가 다를 수 있지만
    // 최대 1px 차이이고 정확도 기준을 넘기므로 의도적으로 받아들인다.
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
