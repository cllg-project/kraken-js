'use strict';
const { test, describe, before } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { KrakenSegmenter } = require('../src/segmenter');

const MODEL   = path.join(__dirname, 'fixtures/segmentation.js_mlmodel');
const FULLPAGE = path.join(__dirname, 'fixtures/fullpage.png');

let segmenter;
let result;

before(async () => {
  segmenter = await KrakenSegmenter.create(MODEL);
  result = await segmenter.segment(FULLPAGE);
});

// ---------------------------------------------------------------------------
// KrakenSegmenter.create
// ---------------------------------------------------------------------------

describe('KrakenSegmenter.create', () => {
  test('returns a KrakenSegmenter instance', () => {
    assert.ok(segmenter instanceof KrakenSegmenter);
  });

  test('metadata has required keys', () => {
    const m = segmenter._meta;
    assert.ok('model_type' in m);
    assert.ok('height' in m);
    assert.ok('channels' in m);
    assert.ok('class_mapping' in m);
  });

  test('metadata height is 1800', () => {
    assert.equal(segmenter._meta.height, 1800);
  });

  test('metadata channels is 3', () => {
    assert.equal(segmenter._meta.channels, 3);
  });

  test('class_mapping has baselines group', () => {
    assert.ok('baselines' in segmenter._meta.class_mapping);
  });

  test('class_mapping baselines has DefaultLine', () => {
    assert.ok('DefaultLine' in segmenter._meta.class_mapping.baselines);
  });

  test('throws on non-existent model file', async () => {
    await assert.rejects(() => KrakenSegmenter.create('/no/such/model.js_mlmodel'));
  });
});

// ---------------------------------------------------------------------------
// segment — result shape
// ---------------------------------------------------------------------------

describe('segment result shape', () => {
  test('returns object with lines and imageSize', () => {
    assert.ok('lines' in result);
    assert.ok('imageSize' in result);
  });

  test('imageSize matches fullpage.png dimensions', () => {
    assert.equal(result.imageSize.width, 2479);
    assert.equal(result.imageSize.height, 3508);
  });

  test('lines is a non-empty array', () => {
    assert.ok(Array.isArray(result.lines));
    assert.ok(result.lines.length > 0, `expected lines, got ${result.lines.length}`);
  });

  test('plausible number of lines for a manuscript page (10–150)', () => {
    assert.ok(result.lines.length >= 10, `too few lines: ${result.lines.length}`);
    assert.ok(result.lines.length <= 150, `too many lines: ${result.lines.length}`);
  });
});

// ---------------------------------------------------------------------------
// segment — line structure
// ---------------------------------------------------------------------------

describe('segment line structure', () => {
  test('each line has obb and type', () => {
    for (const line of result.lines) {
      assert.ok('obb' in line, 'missing obb');
      assert.ok('type' in line, 'missing type');
    }
  });

  test('each obb has required fields', () => {
    for (const { obb } of result.lines) {
      for (const key of ['cx', 'cy', 'w', 'h', 'angle', 'corners']) {
        assert.ok(key in obb, `missing obb.${key}`);
      }
    }
  });

  test('each obb has 4 corners', () => {
    for (const { obb } of result.lines) {
      assert.equal(obb.corners.length, 4);
    }
  });

  test('each corner is [x, y] within image bounds', () => {
    const { width, height } = result.imageSize;
    for (const { obb } of result.lines) {
      for (const [x, y] of obb.corners) {
        assert.ok(x >= 0 && x < width,  `corner x=${x} out of [0, ${width})`);
        assert.ok(y >= 0 && y < height, `corner y=${y} out of [0, ${height})`);
      }
    }
  });

  test('angle is in (-π/2, π/2]', () => {
    for (const { obb } of result.lines) {
      assert.ok(obb.angle > -Math.PI / 2, `angle too small: ${obb.angle}`);
      assert.ok(obb.angle <= Math.PI / 2,  `angle too large: ${obb.angle}`);
    }
  });

  test('type is DefaultLine or DefaultLine-Margin', () => {
    const valid = new Set(['DefaultLine', 'DefaultLine-Margin']);
    for (const { type } of result.lines) {
      assert.ok(valid.has(type), `unexpected type: ${type}`);
    }
  });
});

// ---------------------------------------------------------------------------
// segment — reading order
// ---------------------------------------------------------------------------

describe('segment reading order', () => {
  test('lines are sorted by cy ascending', () => {
    for (let i = 1; i < result.lines.length; i++) {
      assert.ok(
        result.lines[i].obb.cy >= result.lines[i - 1].obb.cy,
        `line ${i} cy=${result.lines[i].obb.cy} < previous cy=${result.lines[i - 1].obb.cy}`
      );
    }
  });
});
