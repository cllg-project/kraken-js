'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { MultiRecognizer } = require('../src/multirecognizer');

const chars = (s, x0 = 0) => [...s].map((char, i) => ({ char, conf: 0.99, x0: x0 + i * 10, x1: x0 + i * 10 + 8 }));
const stub = (text, height = 64, calls) => ({
  _meta: { height },
  async recognize() { calls && calls.push(text); return { text, chars: chars(text) }; },
});

test('rejects a missing role', () => {
  assert.throws(() => new MultiRecognizer({ base: stub('a'), new: stub('a') }), TypeError);
});

test('keeps base text, merges agreed sign, exposes model readings', async () => {
  const calls = [];
  const m = new MultiRecognizer({
    base: stub('abxd', 64, calls),
    new: stub('ab[d', 64, calls),
    spec: stub('__[_', 64, calls),
  });
  // base spike 'x' has conf 0.99 and is not confusable -> conflict, text unchanged
  const r = await m.recognize(Buffer.alloc(0));
  assert.strictEqual(calls.length, 3);
  assert.strictEqual(r.text, 'abxd');
  assert.deepStrictEqual(r.models, { base: 'abxd', new: 'ab[d', spec: '__[_' });
  assert.strictEqual(r.edits[0].conflict, true);
});

test('x of a model with another height is rescaled to the base scale', async () => {
  // half-height models: their x are half the base x, so the '[' at base x 10 is read at 5
  const half = text => ({
    _meta: { height: 32 },
    async recognize() { return { text, chars: [...text].map((char, i) => ({ char, conf: 0.99, x0: i * 5, x1: i * 5 + 4 })) }; },
  });
  const m = new MultiRecognizer({ base: stub('a;c', 64), new: half('a[c'), spec: half('_[_') });
  const r = await m.recognize(Buffer.alloc(0));
  assert.strictEqual(r.text, 'a[c');
});

// ---------------------------------------------------------------------------
// Real models (about 120 MB, not committed): `npm run fixtures:multimodel`
// ---------------------------------------------------------------------------
const fs = require('node:fs');
const path = require('node:path');
const DIR = path.join(__dirname, 'fixtures/multimodel');
const PATHS = Object.fromEntries(['base', 'new', 'spec'].map(r => [r, path.join(DIR, `${r}.js_mlmodel`)]));
const HAVE = Object.values(PATHS).every(p => fs.existsSync(p));
const LINE = path.join(__dirname, 'fixtures/example_line.png');

test('real models: base text kept, per-model readings returned',
  { skip: HAVE ? false : 'run `npm run fixtures:multimodel` to enable' }, async () => {
    const m = await MultiRecognizer.create(PATHS, { threads: 3 });
    const r = await m.recognize(LINE);
    assert.ok(r.text.length > 0);
    assert.deepStrictEqual(Object.keys(r.models), ['base', 'new', 'spec']);
    assert.strictEqual(r.models.base.normalize('NFD').includes('ἀλλὰ ἀλόγῳ'.normalize('NFD')), true);
    assert.match(r.models.spec, /_/);            // the letter-masked reader emits "_" for letters
    assert.ok(Array.isArray(r.edits) && Array.isArray(r.candidates));
    assert.strictEqual(r.chars.map(c => c.char).join(''), r.text); // chars line up with the merged text
  });

test('chars follow the merged text: inserted sign gets a span, substituted char its confidence', async () => {
  const lowBase = {
    _meta: { height: 64 },
    async recognize() { return { text: 'ab;d', chars: [['a', .99], ['b', .99], [';', .3], ['d', .99]].map(([char, conf], i) => ({ char, conf, x0: i * 10, x1: i * 10 + 8 })) }; },
  };
  const sign = text => ({ _meta: { height: 64 }, async recognize() { return { text, chars: [...text].map((char, i) => ({ char, conf: .98, x0: i * 10, x1: i * 10 + 8 })) }; } });
  const m = new MultiRecognizer({ base: lowBase, new: sign('ab[d'), spec: sign('__[_') });
  const r = await m.recognize(Buffer.alloc(0));
  assert.strictEqual(r.text, 'ab[d');
  assert.deepStrictEqual(r.chars.map(c => c.char), ['a', 'b', '[', 'd']);
  assert.strictEqual(r.chars[2].conf, 0.98);
  assert.deepStrictEqual([r.chars[2].x0, r.chars[2].x1], [20, 28]); // span of the substituted base char
});
