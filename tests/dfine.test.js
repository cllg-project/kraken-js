'use strict';
const { test, describe, before } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const sharp = require('sharp');
const ort = require('onnxruntime-node');
const { DFineSegmenter, selectDetections } = require('../src/dfine');
const { KrakenSegmenter } = require('../src/segmenter');
const { KrakenPipeline } = require('../src/pipeline');
const { KrakenRecognizer } = require('../src/recognizer');
const { loadJsMlmodel } = require('../src/loader');
const { preprocessDetectionImage } = require('../src/preprocess');

const MODEL     = path.join(__dirname, 'fixtures/dfine.js_mlmodel');
const REC_MODEL = path.join(__dirname, 'fixtures/model_best.js_mlmodel');
const FULLPAGE  = path.join(__dirname, 'fixtures/fullpage.png');
const EXAMPLE   = path.join(__dirname, 'fixtures/example_line.png');

// The fixture is the LADaS nano layout model, exported with:
//   env/bin/python3 export_kraken_onnx.py ladas_n.safetensors tests/fixtures/dfine.js_mlmodel
// The expected boxes below are what dfine-kraken's own `predict` returns for it.

/**
 * Build a stubbed ONNX session returning fixed D-FINE outputs.
 * `dets` is a list of {query, label, score, box: [cx, cy, w, h]} (normalized).
 */
function stubSession(Q, C, dets) {
  const scores = new Float32Array(Q * C);
  const boxes = new Float32Array(Q * 4);
  for (const { query, label, score, box } of dets) {
    scores[query * C + label] = score;
    boxes.set(box, query * 4);
  }
  return {
    async run() {
      return {
        scores: new ort.Tensor('float32', scores, [1, Q, C]),
        boxes: new ort.Tensor('float32', boxes, [1, Q, 4]),
      };
    },
  };
}

const STUB_META = {
  model_type: 'segmentation',
  architecture: 'dfine',
  variant: 'nano',
  image_size: [64, 64],
  channels: 3,
  num_classes: 5,
  num_top_queries: 300,
  class_mapping: {
    lines: { DefaultLine: 1, 'DefaultLine-Alias': 1, HeadingLine: 2 },
    regions: { MainZone: 3, MarginTextZone: 4 },
  },
};

// ---------------------------------------------------------------------------
// selectDetections
// ---------------------------------------------------------------------------

describe('selectDetections', () => {
  const Q = 3, C = 2;
  // query 0: class 1 = 0.9, query 1: class 0 = 0.6 and class 1 = 0.55, query 2: 0.4
  const scores = Float32Array.from([0.1, 0.9, 0.6, 0.55, 0.4, 0.0]);
  const boxes = Float32Array.from([
    0.5, 0.5, 0.2, 0.4,
    0.25, 0.25, 0.5, 0.5,
    0.9, 0.9, 0.4, 0.4,
  ]);
  const opts = { topK: 300, scoreThreshold: 0.5, width: 200, height: 100 };

  test('keeps (query, class) pairs above threshold by descending score', () => {
    const dets = selectDetections(scores, boxes, Q, C, opts);
    assert.deepEqual(dets.map(d => d.label), [1, 0, 1]);
    assert.deepEqual(dets.map(d => +d.score.toFixed(2)), [0.9, 0.6, 0.55]);
  });

  test('a query can yield one detection per class', () => {
    const dets = selectDetections(scores, boxes, Q, C, opts);
    assert.deepEqual(dets[1].bbox, dets[2].bbox);
  });

  test('converts normalized cxcywh to pixel xyxy', () => {
    const [first] = selectDetections(scores, boxes, Q, C, opts);
    assert.deepEqual(first.bbox.map(v => +v.toFixed(3)), [80, 30, 120, 70]);
  });

  test('clamps boxes to the image', () => {
    const dets = selectDetections(scores, boxes, Q, C, { ...opts, scoreThreshold: 0.3 });
    const last = dets[dets.length - 1];
    assert.deepEqual(last.bbox.map(v => +v.toFixed(3)), [140, 70, 200, 100]);
  });

  test('topK limits the number of detections', () => {
    assert.equal(selectDetections(scores, boxes, Q, C, { ...opts, topK: 2 }).length, 2);
  });
});

// ---------------------------------------------------------------------------
// preprocessDetectionImage
// ---------------------------------------------------------------------------

describe('preprocessDetectionImage', () => {
  test('resizes to exactly image_size, 3-channel CHW in [0, 1], not inverted', async () => {
    const white = await sharp({ create: { width: 30, height: 50, channels: 4, background: '#fff' } })
      .png().toBuffer();
    const { data, width, height } = await preprocessDetectionImage(white, { image_size: [40, 20] });
    assert.equal(width, 20);
    assert.equal(height, 40);
    assert.equal(data.length, 3 * 40 * 20);
    assert.ok(data.every(v => v === 1));
  });

  test('expands grayscale input to RGB', async () => {
    const gray = await sharp({ create: { width: 8, height: 8, channels: 3, background: '#808080' } })
      .grayscale().png().toBuffer();
    const { data } = await preprocessDetectionImage(gray, { image_size: [8, 8] });
    assert.equal(data.length, 3 * 64);
    assert.ok(Math.abs(data[0] - 128 / 255) < 1e-6);
  });
});

// ---------------------------------------------------------------------------
// DFineSegmenter with a stubbed session (line and region post-processing)
// ---------------------------------------------------------------------------

describe('DFineSegmenter post-processing', () => {
  let result;
  const page = sharp({ create: { width: 200, height: 100, channels: 3, background: '#fff' } }).png().toBuffer();

  before(async () => {
    const session = stubSession(6, 5, [
      { query: 0, label: 3, score: 0.95, box: [0.3, 0.5, 0.5, 0.9] },    // MainZone, left
      { query: 1, label: 4, score: 0.80, box: [0.8, 0.5, 0.3, 0.9] },    // MarginTextZone, right
      { query: 2, label: 1, score: 0.90, box: [0.3, 0.7, 0.4, 0.1] },    // DefaultLine, lower
      { query: 3, label: 2, score: 0.70, box: [0.3, 0.2, 0.4, 0.1] },    // HeadingLine, upper
      { query: 4, label: 1, score: 0.60, box: [0.8, 0.45, 0.2, 0.1] },   // DefaultLine in margin
      { query: 5, label: 0, score: 0.99, box: [0.5, 0.5, 1.0, 1.0] },    // unmapped class 0
    ]);
    const seg = new DFineSegmenter(session, STUB_META, {});
    result = await seg.segment(await page);
  });

  test('reports the original image size', () => {
    assert.deepEqual(result.imageSize, { width: 200, height: 100 });
  });

  test('drops classes absent from the class mapping', () => {
    assert.equal(result.lines.length + result.regions.length, 5);
  });

  test('splits detections into typed regions and lines', () => {
    assert.deepEqual(result.regions.map(r => r.type).sort(), ['MainZone', 'MarginTextZone']);
    assert.deepEqual(result.lines.map(l => l.type).sort(), ['DefaultLine', 'DefaultLine', 'HeadingLine']);
  });

  test('uses the first name when several map to one class', () => {
    assert.ok(!result.lines.some(l => l.type === 'DefaultLine-Alias'));
  });

  test('lines are in reading order', () => {
    const cys = result.lines.map(l => l.obb.cy);
    assert.deepEqual(cys, [...cys].sort((a, b) => a - b));
    assert.equal(result.lines[0].type, 'HeadingLine');
  });

  test('line obb describes the box', () => {
    const line = result.lines.find(l => l.type === 'HeadingLine');
    assert.deepEqual(line.bbox.map(v => +v.toFixed(3)), [20, 15, 100, 25]);
    const { cx, cy, w, h, angle } = line.obb;
    assert.deepEqual([cx, cy, w, h].map(v => +v.toFixed(3)), [60, 20, 80, 10]);
    assert.equal(angle, 0);
    assert.deepEqual(line.polygon.map(p => p.map(v => +v.toFixed(3))), [[20, 15], [100, 15], [100, 25], [20, 25]]);
  });

  test('lines are linked to the regions containing their centre', () => {
    const main = result.regions.findIndex(r => r.type === 'MainZone');
    const margin = result.regions.findIndex(r => r.type === 'MarginTextZone');
    const [heading, body, marginLine] = [
      result.lines.find(l => l.type === 'HeadingLine'),
      result.lines.find(l => l.type === 'DefaultLine' && l.obb.cx < 100),
      result.lines.find(l => l.type === 'DefaultLine' && l.obb.cx > 100),
    ];
    assert.deepEqual(heading.regions, [main]);
    assert.deepEqual(body.regions, [main]);
    assert.deepEqual(marginLine.regions, [margin]);
  });

  test('scoreThreshold is honoured', async () => {
    const session = stubSession(2, 5, [
      { query: 0, label: 1, score: 0.9, box: [0.5, 0.5, 0.2, 0.2] },
      { query: 1, label: 1, score: 0.4, box: [0.5, 0.2, 0.2, 0.2] },
    ]);
    const low = await new DFineSegmenter(session, STUB_META, { scoreThreshold: 0.3 }).segment(await page);
    const def = await new DFineSegmenter(session, STUB_META, {}).segment(await page);
    assert.equal(low.lines.length, 2);
    assert.equal(def.lines.length, 1);
  });
});

// ---------------------------------------------------------------------------
// Pipeline with D-FINE line boxes
// ---------------------------------------------------------------------------

describe('KrakenPipeline with D-FINE lines', () => {
  test('recognizes the detected box as-is', async () => {
    const line = await sharp(EXAMPLE).metadata();
    // Put the example line on a larger white page, 40 px from the top-left.
    const page = await sharp({
      create: { width: line.width + 80, height: line.height + 200, channels: 3, background: '#fff' },
    }).composite([{ input: EXAMPLE, left: 40, top: 40 }]).png().toBuffer();
    const pageW = line.width + 80, pageH = line.height + 200;

    const session = stubSession(1, 5, [{
      query: 0, label: 1, score: 0.9,
      box: [(40 + line.width / 2) / pageW, (40 + line.height / 2) / pageH, line.width / pageW, line.height / pageH],
    }]);
    const segmenter = new DFineSegmenter(session, STUB_META, {});
    const recognizer = await KrakenRecognizer.create(REC_MODEL);
    const pipeline = new KrakenPipeline(segmenter, recognizer, {});

    const [res] = await pipeline.process(page);
    const direct = await recognizer.recognize(EXAMPLE);
    assert.equal(res.type, 'DefaultLine');
    assert.deepEqual(res.polygon.map(p => p.map(Math.round)),
      [[40, 40], [40 + line.width, 40], [40 + line.width, 40 + line.height], [40, 40 + line.height]]);
    assert.equal(res.text, direct.text);
  });
});

// ---------------------------------------------------------------------------
// Real model (skipped without the fixture)
// ---------------------------------------------------------------------------

describe('D-FINE model', () => {
  let seg, result;

  before(async () => {
    seg = await KrakenSegmenter.create(MODEL);
    result = await seg.segment(FULLPAGE);
  });

  test('metadata describes a D-FINE segmentation model', () => {
    const { metadata } = loadJsMlmodel(MODEL);
    assert.equal(metadata.model_type, 'segmentation');
    assert.equal(metadata.architecture, 'dfine');
    assert.equal(metadata.variant, 'nano');
    assert.deepEqual(metadata.image_size, [1280, 1280]);
    assert.ok('MainZone-P' in metadata.class_mapping.regions);
    assert.deepEqual(metadata.class_mapping.lines, {});
  });

  test('KrakenSegmenter.create dispatches to DFineSegmenter', () => {
    assert.ok(seg instanceof DFineSegmenter);
  });

  test('graph takes one fixed-size input and returns scores and boxes', () => {
    assert.deepEqual(seg.session.inputNames, ['input']);
    assert.deepEqual(seg.session.outputNames, ['scores', 'boxes']);
  });

  test('detects the same regions as dfine-kraken', () => {
    // dfine-kraken predict() on fullpage.png (boxes rounded to integers)
    const expected = [
      ['RunningTitleZone', [1158, 284, 1478, 368]],
      ['MainZone-P', [458, 814, 2214, 2105]],
      ['MainZone-P', [525, 2465, 2237, 2707]],
      ['MainZone-P', [487, 820, 2212, 2117]],
      ['NumberingZone', [408, 429, 531, 506]],
      ['NumberingZone', [398, 282, 499, 368]],
      ['MarginTextZone-Notes', [520, 2750, 2236, 2886]],
    ];
    assert.equal(result.regions.length, expected.length);
    for (const [type, box] of expected) {
      const match = result.regions.find(r => r.type === type &&
        r.bbox.every((v, i) => Math.abs(v - box[i]) <= 2));
      assert.ok(match, `missing ${type} ${box}`);
    }
  });

  test('a region-only model yields no lines', () => {
    assert.equal(result.lines.length, 0);
  });

  test('regions are sorted top-to-bottom', () => {
    const ys = result.regions.map(r => r.bbox[1]);
    assert.deepEqual(ys, [...ys].sort((a, b) => a - b));
  });
});
