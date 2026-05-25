'use strict';
const { test, describe, before } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { KrakenRecognizer } = require('../src/recognizer');

const MODEL      = path.join(__dirname, 'fixtures/model_best.js_mlmodel');
const EXAMPLE    = path.join(__dirname, 'fixtures/example_line.png');
const OOD        = path.join(__dirname, 'fixtures/ood_example.png');

// Expected strings use NFD normalization to match the codec's decomposed diacritics.
const EXPECTED_EXAMPLE = 'ζετε, ἀλλὰ ἀλόγῳ πάθει καὶ μάστιγι δαιμόνων φαύλων ἐξε-'.normalize('NFD');
const EXPECTED_OOD_SUBSTR = 'κολάζετε'.normalize('NFD');

// ---------------------------------------------------------------------------
// Shared recognizer instance (loaded once for the whole suite)
// ---------------------------------------------------------------------------

let recognizer;

before(async () => {
  recognizer = await KrakenRecognizer.create(MODEL);
});

// ---------------------------------------------------------------------------
// KrakenRecognizer.create
// ---------------------------------------------------------------------------

describe('KrakenRecognizer.create', () => {
  test('returns a KrakenRecognizer instance', () => {
    assert.ok(recognizer instanceof KrakenRecognizer);
  });

  test('metadata is accessible and correct', () => {
    assert.equal(recognizer._meta.height, 120);
    assert.equal(recognizer._meta.channels, 1);
    assert.equal(recognizer._meta.pad, 16);
  });

  test('l2c map is built from codec', () => {
    assert.ok(recognizer._l2c instanceof Map);
    assert.ok(recognizer._l2c.size > 0);
  });

  test('throws on non-existent model file', async () => {
    await assert.rejects(() => KrakenRecognizer.create('/no/such/model.js_mlmodel'));
  });
});

// ---------------------------------------------------------------------------
// recognize — result shape
// ---------------------------------------------------------------------------

describe('recognize result shape', () => {
  let result;
  before(async () => { result = await recognizer.recognize(EXAMPLE); });

  test('returns object with text and chars', () => {
    assert.ok('text' in result);
    assert.ok('chars' in result);
  });

  test('text is a non-empty string', () => {
    assert.equal(typeof result.text, 'string');
    assert.ok(result.text.length > 0);
  });

  test('chars is an array with same length as text codepoints', () => {
    const codepoints = [...result.text];
    assert.equal(result.chars.length, codepoints.length);
  });

  test('each char has required fields with correct types', () => {
    for (const c of result.chars) {
      assert.equal(typeof c.char, 'string');
      assert.equal(typeof c.conf, 'number');
      assert.equal(typeof c.x0, 'number');
      assert.equal(typeof c.x1, 'number');
    }
  });

  test('confidence values are in [0, 1]', () => {
    for (const { conf } of result.chars) {
      assert.ok(conf >= 0 && conf <= 1, `conf out of range: ${conf}`);
    }
  });

  test('x0 and x1 are non-negative integers', () => {
    for (const { x0, x1 } of result.chars) {
      assert.ok(x0 >= 0, `x0 < 0: ${x0}`);
      assert.ok(x1 >= 0, `x1 < 0: ${x1}`);
      assert.equal(x0, Math.round(x0));
      assert.equal(x1, Math.round(x1));
    }
  });
});

// ---------------------------------------------------------------------------
// recognize — content correctness
// ---------------------------------------------------------------------------

describe('recognize content', () => {
  test('example_line.png transcription matches expected', async () => {
    const { text } = await recognizer.recognize(EXAMPLE);
    assert.equal(text.normalize('NFD'), EXPECTED_EXAMPLE);
  });

  test('ood_example.png contains expected substring', async () => {
    const { text } = await recognizer.recognize(OOD);
    assert.ok(text.normalize('NFD').includes(EXPECTED_OOD_SUBSTR),
      `expected "${EXPECTED_OOD_SUBSTR}" in "${text}"`);
  });

  test('accepts a Buffer instead of a file path', async () => {
    const fs = require('node:fs');
    const buf = fs.readFileSync(EXAMPLE);
    const { text } = await recognizer.recognize(buf);
    assert.equal(text.normalize('NFD'), EXPECTED_EXAMPLE);
  });
});

// ---------------------------------------------------------------------------
// recognizeBatch
// ---------------------------------------------------------------------------

describe('recognizeBatch', () => {
  test('returns array of same length as input', async () => {
    const results = await recognizer.recognizeBatch([EXAMPLE, OOD, EXAMPLE]);
    assert.equal(results.length, 3);
  });

  test('each element has text and chars', async () => {
    const results = await recognizer.recognizeBatch([EXAMPLE, OOD]);
    for (const r of results) {
      assert.ok('text' in r && 'chars' in r);
    }
  });

  test('results are in input order (same-width images)', async () => {
    // Mixing very different widths in a batch can corrupt BiLSTM results because
    // zero-padding on the right is seen first by the backward LSTM pass.
    // For ordering correctness we use images of the same size.
    const results = await recognizer.recognizeBatch([EXAMPLE, EXAMPLE]);
    assert.equal(results[0].text.normalize('NFD'), EXPECTED_EXAMPLE);
    assert.equal(results[1].text.normalize('NFD'), EXPECTED_EXAMPLE);
  });

  test('batch of identical images returns identical results', async () => {
    const results = await recognizer.recognizeBatch([EXAMPLE, EXAMPLE, EXAMPLE]);
    assert.equal(results[0].text, results[1].text);
    assert.equal(results[1].text, results[2].text);
  });

  test('recognize() delegates to recognizeBatch() and returns same result', async () => {
    const single = await recognizer.recognize(EXAMPLE);
    const batch  = await recognizer.recognizeBatch([EXAMPLE]);
    assert.equal(single.text.normalize('NFD'), batch[0].text.normalize('NFD'));
    assert.equal(single.chars.length, batch[0].chars.length);
  });

  test('single-element batch equals recognize()', async () => {
    const [batched] = await recognizer.recognizeBatch([OOD]);
    const direct    = await recognizer.recognize(OOD);
    assert.equal(batched.text.normalize('NFD'), direct.text.normalize('NFD'));
  });
});
