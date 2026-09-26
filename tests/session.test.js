'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const { buildSessionOptions } = require('../src/session');

describe('buildSessionOptions', () => {
  test('defaults to CPU and nothing else (ONNX Runtime defaults apply)', () => {
    assert.deepEqual(buildSessionOptions(), { executionProviders: ['cpu'] });
    assert.deepEqual(buildSessionOptions({}), { executionProviders: ['cpu'] });
  });

  test('keeps executionProviders', () => {
    assert.deepEqual(buildSessionOptions({ executionProviders: ['directml', 'cpu'] }),
      { executionProviders: ['directml', 'cpu'] });
  });

  test('threads sets intra-op threads and a single inter-op thread', () => {
    const o = buildSessionOptions({ threads: 4 });
    assert.equal(o.intraOpNumThreads, 4);
    assert.equal(o.interOpNumThreads, 1);
  });

  test('rejects invalid threads', () => {
    for (const bad of [0, -1, 1.5, '4', NaN]) {
      assert.throws(() => buildSessionOptions({ threads: bad }), TypeError);
    }
  });

  test('allowSpinning maps to the ONNX Runtime session config keys', () => {
    assert.deepEqual(buildSessionOptions({ allowSpinning: false }).extra, {
      session: { intra_op: { allow_spinning: '0' }, inter_op: { allow_spinning: '0' } },
    });
    assert.equal(buildSessionOptions({ allowSpinning: true }).extra.session.intra_op.allow_spinning, '1');
  });

  test('raw sessionOptions override derived values and merge extra', () => {
    const o = buildSessionOptions({
      threads: 4,
      allowSpinning: false,
      sessionOptions: {
        intraOpNumThreads: 2,
        graphOptimizationLevel: 'basic',
        extra: { session: { intra_op: { allow_spinning: '1' } }, optimization: { x: '1' } },
      },
    });
    assert.equal(o.intraOpNumThreads, 2);
    assert.equal(o.interOpNumThreads, 1);
    assert.equal(o.graphOptimizationLevel, 'basic');
    assert.deepEqual(o.extra, {
      session: { intra_op: { allow_spinning: '1' }, inter_op: { allow_spinning: '0' } },
      optimization: { x: '1' },
    });
  });
});
