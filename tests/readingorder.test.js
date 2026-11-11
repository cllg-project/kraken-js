'use strict';
const { test, before } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { ReadingOrder } = require('../src/readingorder');

// Reference orders produced by Kraken's own `neural_reading_order` (ROMLP) on the same
// lines, see the commit that added this file for the generator.
const MODEL = path.join(__dirname, 'fixtures/reading_order.js_mlmodel');
const CASES = require('./fixtures/reading_order_cases.json');

let ro;
before(async () => { ro = await ReadingOrder.create(MODEL); });

test('matches Kraken ROMLP ordering', async () => {
  let same = 0;
  for (const c of CASES) {
    const got = await ro.order(c.lines, c.imageSize);
    if (JSON.stringify(got) === JSON.stringify(c.order)) same++;
  }
  // float32 (torch) vs float64 decoding can flip a near-tie; allow a couple at most
  assert.equal(same, CASES.length);
});

test('0 and 1 lines', async () => {
  assert.deepEqual(await ro.order([], { width: 10, height: 10 }), []);
  assert.deepEqual(await ro.order([{ obb: { cx: 1, cy: 1, w: 2, angle: 0 }, type: 'DefaultLine' }], { width: 10, height: 10 }), [0]);
});
