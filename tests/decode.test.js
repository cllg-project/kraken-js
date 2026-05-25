'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const { buildL2C, greedyCTC, decodeCodec } = require('../src/decode');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a logits array (C × W, layout c*W+t) where the given class is
 * dominant at the given timestep (logit=100, all others=0).
 */
function makeLogits(C, W, dominantAt = []) {
  const logits = new Float32Array(C * W);
  for (const [t, c] of dominantAt) logits[c * W + t] = 100;
  return logits;
}

// ---------------------------------------------------------------------------
// buildL2C
// ---------------------------------------------------------------------------

describe('buildL2C', () => {
  test('single-label entries', () => {
    const l2c = buildL2C({ a: [1], b: [2], c: [3] });
    assert.equal(l2c.get('1'), 'a');
    assert.equal(l2c.get('2'), 'b');
    assert.equal(l2c.get('3'), 'c');
  });

  test('multi-label (ligature) entry', () => {
    const l2c = buildL2C({ æ: [4, 5], a: [1] });
    assert.equal(l2c.get('4,5'), 'æ');
    assert.equal(l2c.get('1'), 'a');
    assert.equal(l2c.has('4'), false); // partial key not present
  });

  test('empty codec produces empty map', () => {
    const l2c = buildL2C({});
    assert.equal(l2c.size, 0);
  });

  test('unicode characters are preserved', () => {
    const l2c = buildL2C({ ζ: [10], '̓': [20] });
    assert.equal(l2c.get('10'), 'ζ');
    assert.equal(l2c.get('20'), '̓');
  });
});

// ---------------------------------------------------------------------------
// greedyCTC
// ---------------------------------------------------------------------------

describe('greedyCTC', () => {
  test('all-blank input returns empty array', () => {
    // All logits = 0 → softmax uniform → argmax picks class 0 (blank) in all ties
    const logits = new Float32Array(3 * 4); // C=3, W=4, all zeros
    assert.deepEqual(greedyCTC(logits, 3, 4), []);
  });

  test('single character at one timestep', () => {
    // C=3, W=5: class 1 dominant only at t=2
    const logits = makeLogits(3, 5, [[2, 1]]);
    const result = greedyCTC(logits, 3, 5);
    assert.equal(result.length, 1);
    assert.equal(result[0].label, 1);
    assert.equal(result[0].t0, 2);
    assert.equal(result[0].t1, 2);
    assert.ok(result[0].conf > 0.99, `conf should be ~1, got ${result[0].conf}`);
  });

  test('consecutive same labels collapse into one run', () => {
    // class 1 at t=0,1,2 (no blank between) → single entry spanning all three
    const logits = makeLogits(3, 4, [[0, 1], [1, 1], [2, 1]]);
    const result = greedyCTC(logits, 3, 4);
    assert.equal(result.length, 1);
    assert.equal(result[0].label, 1);
    assert.equal(result[0].t0, 0);
    assert.equal(result[0].t1, 2);
  });

  test('same label separated by blank produces two entries', () => {
    // [class1, blank, class1, blank]
    const logits = makeLogits(3, 4, [[0, 1], [1, 0], [2, 1]]);
    const result = greedyCTC(logits, 3, 4);
    assert.equal(result.length, 2);
    assert.equal(result[0].label, 1);
    assert.equal(result[0].t0, 0);
    assert.equal(result[0].t1, 0);
    assert.equal(result[1].label, 1);
    assert.equal(result[1].t0, 2);
    assert.equal(result[1].t1, 2);
  });

  test('two different chars separated by blank', () => {
    // [class1, blank, blank, class2, class2]
    const logits = makeLogits(3, 5, [[0, 1], [3, 2], [4, 2]]);
    const result = greedyCTC(logits, 3, 5);
    assert.equal(result.length, 2);
    assert.equal(result[0].label, 1);
    assert.equal(result[0].t0, 0);
    assert.equal(result[0].t1, 0);
    assert.equal(result[1].label, 2);
    assert.equal(result[1].t0, 3);
    assert.equal(result[1].t1, 4);
  });

  test('character at last timestep is flushed', () => {
    const logits = makeLogits(2, 3, [[2, 1]]); // C=2, W=3, class 1 at t=2
    const result = greedyCTC(logits, 2, 3);
    assert.equal(result.length, 1);
    assert.equal(result[0].label, 1);
    assert.equal(result[0].t1, 2);
  });

  test('confidence values are in [0, 1]', () => {
    const logits = makeLogits(4, 6, [[0, 1], [2, 2], [4, 3]]);
    const result = greedyCTC(logits, 4, 6);
    for (const { conf } of result) {
      assert.ok(conf >= 0 && conf <= 1, `conf out of range: ${conf}`);
    }
  });

  test('conf is the maximum softmax value within the run', () => {
    // class 1 at t=0,1 with different logit magnitudes
    const logits = new Float32Array(3 * 3); // C=3, W=3
    logits[1 * 3 + 0] = 10;  // moderate
    logits[1 * 3 + 1] = 100; // very confident
    const result = greedyCTC(logits, 3, 3);
    assert.equal(result.length, 1);
    // conf should reflect the higher-confidence step
    assert.ok(result[0].conf > 0.99);
  });

  test('W=1 single timestep non-blank', () => {
    const logits = makeLogits(2, 1, [[0, 1]]);
    const result = greedyCTC(logits, 2, 1);
    assert.equal(result.length, 1);
    assert.equal(result[0].t0, 0);
    assert.equal(result[0].t1, 0);
  });

  test('W=1 single timestep blank returns empty', () => {
    const logits = new Float32Array(2); // C=2, W=1, all zeros → blank wins
    assert.deepEqual(greedyCTC(logits, 2, 1), []);
  });
});

// ---------------------------------------------------------------------------
// decodeCodec
// ---------------------------------------------------------------------------

describe('decodeCodec', () => {
  const codec = { a: [1], b: [2], æ: [3, 4], z: [5] };
  let l2c;
  test.before = undefined; // no setup hook needed; build inline
  function l2cOf(c) { return buildL2C(c); }

  test('empty label sequence → empty result', () => {
    assert.deepEqual(decodeCodec([], l2cOf(codec)), []);
  });

  test('single-label mapping', () => {
    const labels = [{ label: 1, t0: 0, t1: 0, conf: 0.9 }];
    const result = decodeCodec(labels, l2cOf(codec));
    assert.equal(result.length, 1);
    assert.equal(result[0].char, 'a');
    assert.equal(result[0].conf, 0.9);
    assert.equal(result[0].t0, 0);
    assert.equal(result[0].t1, 0);
  });

  test('multi-label (ligature) match preferred over single', () => {
    // labels [3,4] should decode as "æ" not two separate unknowns
    const labels = [
      { label: 3, t0: 0, t1: 1, conf: 0.8 },
      { label: 4, t0: 2, t1: 3, conf: 0.6 },
    ];
    const result = decodeCodec(labels, l2cOf(codec));
    assert.equal(result.length, 1);
    assert.equal(result[0].char, 'æ');
    assert.equal(result[0].t0, 0);
    assert.equal(result[0].t1, 3);
    // conf = average of group = (0.8 + 0.6) / 2
    assert.ok(Math.abs(result[0].conf - 0.7) < 1e-6);
  });

  test('unknown label is silently skipped', () => {
    const labels = [
      { label: 99, t0: 0, t1: 0, conf: 1 },
      { label: 1,  t0: 1, t1: 1, conf: 1 },
    ];
    const result = decodeCodec(labels, l2cOf(codec));
    assert.equal(result.length, 1);
    assert.equal(result[0].char, 'a');
  });

  test('sequence of multiple chars preserves order', () => {
    const labels = [
      { label: 1, t0: 0, t1: 0, conf: 1 },
      { label: 2, t0: 1, t1: 1, conf: 1 },
      { label: 5, t0: 2, t1: 2, conf: 1 },
    ];
    const result = decodeCodec(labels, l2cOf(codec));
    assert.equal(result.map(c => c.char).join(''), 'abz');
  });

  test('positions span the full group for multi-label chars', () => {
    const labels = [
      { label: 3, t0: 5, t1: 7, conf: 1 },
      { label: 4, t0: 8, t1: 10, conf: 1 },
    ];
    const result = decodeCodec(labels, l2cOf({ æ: [3, 4] }));
    assert.equal(result[0].t0, 5);
    assert.equal(result[0].t1, 10);
  });

  test('greedy: matches longest prefix first', () => {
    // codec has both "3" → "x" and "3,4" → "æ"; should prefer "3,4"
    const mixedCodec = buildL2C({ x: [3], æ: [3, 4] });
    const labels = [
      { label: 3, t0: 0, t1: 0, conf: 1 },
      { label: 4, t0: 1, t1: 1, conf: 1 },
    ];
    const result = decodeCodec(labels, mixedCodec);
    assert.equal(result.length, 1);
    assert.equal(result[0].char, 'æ');
  });

  test('falls back to single-label when multi-label not in codec', () => {
    // label 3 alone is in codec, label 3+4 is not → two separate chars
    const singleOnly = buildL2C({ x: [3], y: [4] });
    const labels = [
      { label: 3, t0: 0, t1: 0, conf: 0.9 },
      { label: 4, t0: 1, t1: 1, conf: 0.8 },
    ];
    const result = decodeCodec(labels, singleOnly);
    assert.equal(result.length, 2);
    assert.equal(result[0].char, 'x');
    assert.equal(result[1].char, 'y');
  });
});
