'use strict';
const { test, describe, before } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { KrakenPipeline } = require('../src/pipeline');

const SEG_MODEL   = path.join(__dirname, 'fixtures/segmentation.js_mlmodel');
const REC_MODEL   = path.join(__dirname, 'fixtures/model_best.js_mlmodel');
const FULLPAGE    = path.join(__dirname, 'fixtures/fullpage.png');
const DOUBLE_PAGE = path.join(__dirname, 'fixtures/double_page.png');

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

// ---------------------------------------------------------------------------
// process — content regression
//
// These tests guard against preprocessing regressions (e.g. missing inversion)
// that silently reduce line-detection quality without breaking structural checks.
// Expected strings are NFD-normalised to match the codec's decomposed diacritics.
// ---------------------------------------------------------------------------

describe('process content regression', () => {
  let fullText;

  before(() => {
    fullText = results.map(r => r.text).join('\n').normalize('NFD');
  });

  // The ALTO ground truth has 39 lines (incl. margin numbers). A correct
  // segmentation yields 35. Polarity-inverted input drops to ~31.
  test('line count matches ALTO segmentation (≥ 33)', () => {
    assert.ok(results.length >= 33,
      `expected ≥ 33 lines, got ${results.length} — possible preprocessing regression`);
  });

  // Lines that only appear when the image polarity is correct
  test('title line Τοῦ ἐν ἀγίοις is detected and contains Πατρὸς', () => {
    assert.ok(fullText.includes('Πατρὸς'.normalize('NFD')),
      'Πατρὸς not found — title line likely missed due to polarity bug');
  });

  test('short line λόγος is detected', () => {
    assert.ok(fullText.includes('λόγος'.normalize('NFD')),
      'λόγος not found — short title lines likely missed');
  });

  test('short line τήν. is detected', () => {
    assert.ok(fullText.includes('τήν'.normalize('NFD')),
      'τήν. not found — isolated short lines likely missed');
  });

  // Stable body-text substrings present across all correct runs
  test('body text contains κηρύγματος', () => {
    assert.ok(fullText.includes('κηρύγματος'.normalize('NFD')),
      'κηρύγματος not found in output');
  });

  test('body text contains εὐρύχωρον', () => {
    assert.ok(fullText.includes('εὐρύχωρον'.normalize('NFD')),
      'εὐρύχωρον not found in output');
  });
});

// ---------------------------------------------------------------------------
// double-page spread — OBB crop + reading order
//
// double_page.png is a landscape two-column spread. Right-page lines are
// angled at −1.4° to −1.8°, exercising the OBB-oriented crop path.
// The column-split heuristic should place the left page first.
// ---------------------------------------------------------------------------

describe('double-page spread', () => {
  let dpResults;

  before(async () => {
    dpResults = await pipeline.process(DOUBLE_PAGE);
  });

  test('detects a plausible number of lines (20–80)', () => {
    assert.ok(dpResults.length >= 20, `too few lines: ${dpResults.length}`);
    assert.ok(dpResults.length <= 80, `too many lines: ${dpResults.length}`);
  });

  // Reading order: left column must precede right column.
  // "κεκλήκασι" appears on the left page; "τετταράκοντα" on the right.
  test('left-page text precedes right-page text (column reading order)', () => {
    const texts = dpResults.map(r => r.text.normalize('NFD'));
    const leftIdx  = texts.findIndex(t => t.includes('κεκλήκασι'.normalize('NFD')));
    const rightIdx = texts.findIndex(t => t.includes('τετταράκοντα'.normalize('NFD')));
    assert.ok(leftIdx  !== -1, '"κεκλήκασι" (left page) not found');
    assert.ok(rightIdx !== -1, '"τετταράκοντα" (right page) not found');
    assert.ok(leftIdx < rightIdx,
      `left-page line (${leftIdx}) should come before right-page line (${rightIdx})`);
  });

  // Right-page body text — these lines are angled and test OBB-oriented cropping.
  test('right-page body contains τετταράκοντα (angled OBB crop)', () => {
    const fullText = dpResults.map(r => r.text).join('\n').normalize('NFD');
    assert.ok(fullText.includes('τετταράκοντα'.normalize('NFD')),
      'τετταράκοντα not found — right-page angled crop likely failing');
  });

  test('right-page body contains ἐλάμβανον (angled OBB crop)', () => {
    const fullText = dpResults.map(r => r.text).join('\n').normalize('NFD');
    assert.ok(fullText.includes('ἐλάμβανον'.normalize('NFD')),
      'ἐλάμβανον not found — right-page angled crop likely failing');
  });

  test('right-page body contains κλινοποιοὺς (angled OBB crop)', () => {
    const fullText = dpResults.map(r => r.text).join('\n').normalize('NFD');
    assert.ok(fullText.includes('κλινοποιοὺς'.normalize('NFD')),
      'κλινοποιοὺς not found — right-page angled crop likely failing');
  });
});
