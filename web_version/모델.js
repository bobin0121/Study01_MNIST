// 내보낸 가중치를 불러와 model.py의 forward와 같은 순서로 순전파한다.
// 드롭아웃은 예측 시 아무 일도 하지 않으므로 구현하지 않는다.

import { conv2d, relu, maxPool2d, linear, softmax } from "./연산.js";

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

// basePath 아래의 가중치정보.json / 가중치.bin을 읽어 모델을 준비한다.
export async function loadModel(basePath) {
  const 매니페스트응답 = await fetch(`${basePath}/가중치정보.json`);
  if (!매니페스트응답.ok) {
    throw new Error(`매니페스트를 불러오지 못했습니다 (${매니페스트응답.status}).`);
  }
  const manifest = await 매니페스트응답.json();

  const 가중치응답 = await fetch(`${basePath}/가중치.bin`);
  if (!가중치응답.ok) {
    throw new Error(`가중치 파일을 불러오지 못했습니다 (${가중치응답.status}).`);
  }
  const buffer = await 가중치응답.arrayBuffer();

  if (buffer.byteLength !== manifest.totalBytes) {
    throw new Error(
      `가중치 크기가 매니페스트와 다릅니다: ${buffer.byteLength} vs ${manifest.totalBytes}`
    );
  }

  // forward()가 실제로 기대하는 shape. 위 상수들로부터 계산해서
  // 상수 하나를 바꾸면 이 표와 forward가 함께 바뀌도록 한다.
  const 기대shape = {
    "conv1.weight": [CONV1_출력, 1, 3, 3],
    "conv1.bias": [CONV1_출력],
    "conv2.weight": [CONV2_출력, CONV1_출력, 3, 3],
    "conv2.bias": [CONV2_출력],
    "fc1.weight": [FC1_출력, CONV2_출력 * (입력한변 / 4) * (입력한변 / 4)],
    "fc1.bias": [FC1_출력],
    "fc2.weight": [분류개수, FC1_출력],
    "fc2.bias": [분류개수],
  };

  const manifest텐서 = {};
  for (const entry of manifest.tensors) {
    manifest텐서[entry.name] = entry;
  }

  for (const [이름, shape] of Object.entries(기대shape)) {
    const entry = manifest텐서[이름];
    if (!entry) {
      throw new Error(`매니페스트에 텐서가 없습니다: ${이름}`);
    }
    const 실제shape = entry.shape;
    const 일치 =
      Array.isArray(실제shape) &&
      실제shape.length === shape.length &&
      실제shape.every((v, i) => v === shape[i]);

    if (!일치) {
      throw new Error(
        `텐서 shape가 맞지 않습니다: ${이름} — 기대 [${shape.join(",")}], ` +
        `실제 [${(실제shape || []).join(",")}]`
      );
    }
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
