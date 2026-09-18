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
