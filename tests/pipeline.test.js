'use strict';
const { test, describe, before } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { KrakenPipeline } = require('../src/pipeline');

const SEG_MODEL  = path.join(__dirname, 'fixtures/segmentation.js_mlmodel');
const REC_MODEL  = path.join(__dirname, 'fixtures/model_best.js_mlmodel');
const FULLPAGE   = path.join(__dirname, 'fixtures/fullpage.png');

let pipeline;
let results;

before(async () => {
  pipeline = await KrakenPipeline.create(SEG_MODEL, REC_MODEL);
  results  = await pipeline.process(FULLPAGE);
});

// ---------------------------------------------------------------------------
// KrakenPipeline.create
// ---------------------------------------------------------------------------

describe('KrakenPipeline.create', () => {
  test('returns a KrakenPipeline instance', () => {
    assert.ok(pipeline instanceof KrakenPipeline);
  });

  test('throws on non-existent segmenter model', async () => {
    await assert.rejects(() => KrakenPipeline.create('/no/seg.js_mlmodel', REC_MODEL));
  });

  test('throws on non-existent recognizer model', async () => {
    await assert.rejects(() => KrakenPipeline.create(SEG_MODEL, '/no/rec.js_mlmodel'));
  });
});

// ---------------------------------------------------------------------------
// process — result shape
// ---------------------------------------------------------------------------

describe('process result shape', () => {
  test('returns a non-empty array', () => {
    assert.ok(Array.isArray(results));
    assert.ok(results.length > 0, `expected results, got ${results.length}`);
  });

  test('plausible number of results for a manuscript page (10–150)', () => {
    assert.ok(results.length >= 10, `too few: ${results.length}`);
    assert.ok(results.length <= 150, `too many: ${results.length}`);
  });

  test('each result has obb, type, text, chars', () => {
    for (const r of results) {
      assert.ok('obb'   in r, 'missing obb');
      assert.ok('type'  in r, 'missing type');
      assert.ok('text'  in r, 'missing text');
      assert.ok('chars' in r, 'missing chars');
    }
  });

  test('text is a string (possibly empty for blank lines)', () => {
    for (const { text } of results) {
      assert.equal(typeof text, 'string');
    }
  });

  test('chars is an array', () => {
    for (const { chars } of results) {
      assert.ok(Array.isArray(chars));
    }
  });

  test('most recognized lines are non-empty', () => {
    const nonEmpty = results.filter(r => r.text.trim().length > 0);
    assert.ok(nonEmpty.length >= results.length * 0.5,
      `fewer than 50% of lines have text: ${nonEmpty.length}/${results.length}`);
  });
});

// ---------------------------------------------------------------------------
// process — OBB correctness
// ---------------------------------------------------------------------------

describe('process OBB correctness', () => {
  test('each obb has required fields', () => {
    for (const { obb } of results) {
      for (const key of ['cx', 'cy', 'w', 'h', 'angle', 'corners']) {
        assert.ok(key in obb, `missing obb.${key}`);
      }
    }
  });

  test('type is DefaultLine or DefaultLine-Margin', () => {
    const valid = new Set(['DefaultLine', 'DefaultLine-Margin']);
    for (const { type } of results) {
      assert.ok(valid.has(type), `unexpected type: ${type}`);
    }
  });
});
