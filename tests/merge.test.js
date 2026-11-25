'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { merge, candidates, spikes, pitch, signTokens, DEFAULT_SIGNS } = require('../src/merge');
const golden = require('./fixtures/merge_golden.json');

const S = (ch, x, conf = 0.99) => ({ ch, x, conf });
const toSp = l => l.map(([ch, x, conf]) => ({ ch, x, conf }));

test('golden parity with the Python reference', () => {
  assert.ok(golden.length > 0);
  golden.forEach((g, i) => {
    const params = { specOnly: false }; // golden output is the Python rule, without the spec-only extension
    for (const [k, v] of Object.entries(g.params)) if (v != null) params[k] = v;
    const r = merge(toSp(g.base), toSp(g.new), toSp(g.spec), params);
    assert.strictEqual(r.text, g.text, `case ${i} text`);
    assert.deepStrictEqual(r.edits, g.edits, `case ${i} edits`);
    assert.deepStrictEqual(r.flags, g.flags, `case ${i} flags`);
    assert.deepStrictEqual(candidates(toSp(g.base), toSp(g.new), r.edits, params), g.candidates, `case ${i} candidates`);
  });
});

test('spikes uses the middle of the span', () => {
  assert.deepStrictEqual(spikes([{ char: 'a', conf: 0.5, x0: 2, x1: 6 }]), [S('a', 4, 0.5)]);
});

test('pitch floors at 4 and defaults to 8', () => {
  assert.strictEqual(pitch([]), 8);
  assert.strictEqual(pitch([S('a', 0), S('b', 1)]), 4);
  assert.strictEqual(pitch([S('a', 0), S('b', 10)]), 10);
});

test('adjacent bars become one || token', () => {
  const t = signTokens([S('|', 10, 0.9), S('|', 14, 0.8)], 8, DEFAULT_SIGNS);
  assert.deepStrictEqual(t, [S('||', 12, 0.8)]);
});

const base = [S('a', 10), S('b', 20, 0.3), S('c', 30), S('d', 40)];

test('sign agreed by new and spec replaces a low-confidence base char', () => {
  const r = merge(base, [S('[', 21)], [S('[', 22)]);
  assert.strictEqual(r.text, 'a[cd');
  assert.strictEqual(r.edits[0].to, '[');
});

test('sign only new reads is not applied, but is a candidate', () => {
  const r = merge(base, [S('[', 21)], []);
  assert.strictEqual(r.text, 'abcd');
  assert.strictEqual(candidates(base, [S('[', 21)], r.edits)[0].sign, '[');
});

test('confident base char is never overwritten: conflict', () => {
  const r = merge(base, [S('{', 30.5)], [S('{', 30.5)]);
  assert.strictEqual(r.text, 'abcd');
  assert.strictEqual(r.edits[0].conflict, true);
});

test('sign fills free space', () => {
  const r = merge(base, [S('†', 35)], [S('†', 35)]);
  assert.strictEqual(r.text, 'abc†d');
});

test('spec-only sign confident in free space is inserted (kind: spec)', () => {
  const r = merge(base, [], [S('<', 25, 0.997)]);
  assert.strictEqual(r.text, 'ab<cd');
  assert.strictEqual(r.edits[0].kind, 'spec');
});

test('spec-only sign below candConf is ignored', () => {
  assert.strictEqual(merge(base, [], [S('<', 25, 0.8)]).text, 'abcd');
});

test('spec-only sign over a confident base char is a conflict, not applied', () => {
  const r = merge(base, [], [S('<', 30.5, 0.99)]);
  assert.strictEqual(r.text, 'abcd');
  assert.strictEqual(r.edits[0].conflict, true);
});

test('specOnly: false restores the two-model agreement rule', () => {
  assert.strictEqual(merge(base, [], [S('<', 25, 0.997)], { specOnly: false }).text, 'abcd');
});
