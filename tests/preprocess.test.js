'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const sharp = require('sharp');
const { preprocessImage, buildBatch, toChw } = require('../src/preprocess');

// ---------------------------------------------------------------------------
// Helpers — synthetic images via sharp
// ---------------------------------------------------------------------------

// sharp.create() only supports channels 3 or 4.
// We always create a 3-channel PNG; preprocessImage converts to grayscale when channels=1.
function solidImage(width, height, _channels, fill) {
  const bg = typeof fill === 'number' ? { r: fill, g: fill, b: fill } : fill;
  return sharp({ create: { width, height, channels: 3, background: bg } })
    .png()
    .toBuffer();
}

// ---------------------------------------------------------------------------
// toChw
// ---------------------------------------------------------------------------

describe('toChw', () => {
  test('channels=1 returns the same Float32Array reference', () => {
    const data = new Float32Array([1, 2, 3, 4]);
    const result = toChw({ data, width: 2, height: 2 }, 1);
    assert.equal(result, data);
  });

  test('channels=3: correct HWC→CHW reorder', () => {
    // 1×2 image (height=1, width=2) with 3 channels
    // HWC pixel layout: [R00,G00,B00, R01,G01,B01]
    const data = new Float32Array([0.1, 0.2, 0.3,  0.4, 0.5, 0.6]);
    const result = toChw({ data, width: 2, height: 1 }, 3);
    // CHW: R channel = [R00, R01], G = [G00, G01], B = [B00, B01]
    assert.ok(Math.abs(result[0] - 0.1) < 1e-6); // R00
    assert.ok(Math.abs(result[1] - 0.4) < 1e-6); // R01
    assert.ok(Math.abs(result[2] - 0.2) < 1e-6); // G00
    assert.ok(Math.abs(result[3] - 0.5) < 1e-6); // G01
    assert.ok(Math.abs(result[4] - 0.3) < 1e-6); // B00
    assert.ok(Math.abs(result[5] - 0.6) < 1e-6); // B01
  });

  test('channels=3: output length is channels*height*width', () => {
    const data = new Float32Array(3 * 4 * 5); // H=4, W=5, C=3 in HWC = 60
    const result = toChw({ data, width: 5, height: 4 }, 3);
    assert.equal(result.length, 3 * 4 * 5);
  });
});

// ---------------------------------------------------------------------------
// buildBatch
// ---------------------------------------------------------------------------

describe('buildBatch', () => {
  function makeTensor(width, height, channels, fill = 1.0) {
    const data = new Float32Array(channels * height * width).fill(fill);
    return { data, width, height };
  }

  test('single tensor: batchData equals input data', () => {
    const t = makeTensor(10, 4, 1, 0.5);
    const { batchData, batchWidth, widths } = buildBatch([t], 1);
    assert.equal(batchWidth, 10);
    assert.deepEqual(widths, [10]);
    assert.equal(batchData.length, 4 * 10);
    for (let i = 0; i < batchData.length; i++) {
      assert.ok(Math.abs(batchData[i] - 0.5) < 1e-6);
    }
  });

  test('two equal-width tensors: batchWidth is that width', () => {
    const t1 = makeTensor(8, 4, 1, 0.3);
    const t2 = makeTensor(8, 4, 1, 0.7);
    const { batchWidth, widths } = buildBatch([t1, t2], 1);
    assert.equal(batchWidth, 8);
    assert.deepEqual(widths, [8, 8]);
  });

  test('shorter tensor is zero-padded on the right', () => {
    const H = 2, C = 1;
    const t1 = makeTensor(6, H, C, 1.0); // wider
    const t2 = makeTensor(4, H, C, 1.0); // narrower — will be padded

    const { batchData, batchWidth } = buildBatch([t1, t2], C);
    assert.equal(batchWidth, 6);

    // Second image's data occupies cols 0-3 of each row; cols 4-5 must be zero
    // batch layout: [N=0: C*H*W, N=1: C*H*W]
    // For N=1, row h=0: positions [1*C*H*W + h*W + 4] and [...+5] should be 0
    const base = 1 * C * H * batchWidth;
    for (let h = 0; h < H; h++) {
      assert.equal(batchData[base + h * batchWidth + 4], 0);
      assert.equal(batchData[base + h * batchWidth + 5], 0);
      assert.equal(batchData[base + h * batchWidth + 0], 1); // filled area
    }
  });

  test('batchWidth is the maximum of all input widths', () => {
    const tensors = [3, 7, 5].map(w => makeTensor(w, 2, 1));
    const { batchWidth } = buildBatch(tensors, 1);
    assert.equal(batchWidth, 7);
  });

  test('widths array matches input tensor widths', () => {
    const tensors = [10, 6, 8].map(w => makeTensor(w, 3, 1));
    const { widths } = buildBatch(tensors, 1);
    assert.deepEqual(widths, [10, 6, 8]);
  });
});

// ---------------------------------------------------------------------------
// preprocessImage
// ---------------------------------------------------------------------------

describe('preprocessImage', () => {
  const HEIGHT = 32, PAD = 8;
  const GRAYSCALE_META = { height: HEIGHT, channels: 1, pad: PAD };
  const RGB_META       = { height: HEIGHT, channels: 3, pad: PAD };

  test('output height matches target height', async () => {
    const img = await solidImage(200, 50, 1, 128);
    const result = await preprocessImage(img, GRAYSCALE_META);
    assert.equal(result.height, HEIGHT);
  });

  test('output width includes 2×pad', async () => {
    const img = await solidImage(100, 50, 1, 128);
    const result = await preprocessImage(img, GRAYSCALE_META);
    // proportional width after resize + 2*pad on sides
    assert.ok(result.width > 2 * PAD);
    // padding contributes exactly 2*pad extra pixels
    const unpaddedWidth = result.width - 2 * PAD;
    assert.ok(unpaddedWidth > 0);
  });

  test('output data length = height × width (channels=1)', async () => {
    const img = await solidImage(200, 40, 1, 128);
    const result = await preprocessImage(img, GRAYSCALE_META);
    assert.equal(result.data.length, result.height * result.width);
  });

  test('output data length = channels × height × width (channels=3)', async () => {
    const img = await solidImage(200, 40, 3, { r: 128, g: 64, b: 32 });
    const result = await preprocessImage(img, RGB_META);
    assert.equal(result.data.length, 3 * result.height * result.width);
  });

  test('all output values are in [0, 1]', async () => {
    const img = await solidImage(150, 30, 1, 100);
    const { data } = await preprocessImage(img, GRAYSCALE_META);
    for (let i = 0; i < data.length; i++) {
      assert.ok(data[i] >= 0 && data[i] <= 1, `value out of range at ${i}: ${data[i]}`);
    }
  });

  test('white pixels (255) become 0 after inversion', async () => {
    const img = await solidImage(50, HEIGHT, 1, 255); // all white
    const { data } = await preprocessImage(img, { ...GRAYSCALE_META, pad: 0 });
    for (let i = 0; i < data.length; i++) {
      assert.ok(Math.abs(data[i]) < 0.01, `expected ~0, got ${data[i]}`);
    }
  });

  test('black pixels (0) become 1 after inversion', async () => {
    const img = await solidImage(50, HEIGHT, 1, 0); // all black
    const { data } = await preprocessImage(img, { ...GRAYSCALE_META, pad: 0 });
    // image area (not padding) should be ~1.0
    const imgWidth = data.length / HEIGHT;
    for (let h = 0; h < HEIGHT; h++) {
      // skip first/last PAD=0 cols (no pad here)
      for (let w = 0; w < imgWidth; w++) {
        assert.ok(Math.abs(data[h * imgWidth + w] - 1.0) < 0.01);
      }
    }
  });

  test('padding columns are 0 (inverted white)', async () => {
    // pad=8: leftmost 8 and rightmost 8 columns should be ~0 (white→inverted)
    const img = await solidImage(100, HEIGHT, 1, 0); // all black image
    const { data, width } = await preprocessImage(img, GRAYSCALE_META);
    // Left pad region: first PAD columns per row should be 0 (white padding, inverted)
    for (let h = 0; h < HEIGHT; h++) {
      for (let w = 0; w < PAD; w++) {
        assert.ok(Math.abs(data[h * width + w]) < 0.01,
          `left pad [h=${h},w=${w}] should be ~0, got ${data[h * width + w]}`);
      }
      for (let w = width - PAD; w < width; w++) {
        assert.ok(Math.abs(data[h * width + w]) < 0.01,
          `right pad [h=${h},w=${w}] should be ~0, got ${data[h * width + w]}`);
      }
    }
  });

  test('accepts a file path', async () => {
    const result = await preprocessImage(path.join(__dirname, 'fixtures/example_line.png'), GRAYSCALE_META);
    assert.equal(result.height, HEIGHT);
    assert.ok(result.width > 2 * PAD);
  });

  test('aspect ratio is preserved (wide image stays wide)', async () => {
    const img = await solidImage(400, 20, 1, 128); // 20:1 aspect
    const result = await preprocessImage(img, GRAYSCALE_META);
    const unpadded = result.width - 2 * PAD;
    // Original aspect: 400/20 = 20. At height=32: expected width ≈ 32*20 = 640
    assert.ok(unpadded > 500, `expected wide image, got width ${unpadded}`);
  });
});
