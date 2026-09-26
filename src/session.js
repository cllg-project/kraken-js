'use strict';

/**
 * Build the ONNX Runtime session options for a segmenter or recognizer from its
 * `create()` options.
 *
 * With none of the options below set, this is exactly `{ executionProviders }` — ONNX
 * Runtime's own defaults apply, i.e. one intra-op thread per physical core, and idle
 * pool threads busy-wait ("spin") between runs. On a many-core machine that keeps every
 * core at 100% for the whole run, although (measured on a 35-line page, 24 logical
 * cores) 4 threads recognize the page about as fast as the default ~20.
 *
 * @param {object}   [opts]
 * @param {string[]} [opts.executionProviders=['cpu']]
 * @param {number}   [opts.threads]        Threads per inference (`intraOpNumThreads`).
 *                                         Also sets `interOpNumThreads: 1`: the graphs run
 *                                         sequentially, so that pool is otherwise idle.
 * @param {boolean}  [opts.allowSpinning]  `false` lets idle pool threads sleep instead of
 *                                         busy-waiting between runs (small latency cost).
 * @param {object}   [opts.sessionOptions] Raw ONNX Runtime session options; override the
 *                                         values derived from `threads`/`allowSpinning`.
 * @returns {object} options for `ort.InferenceSession.create`
 */
function buildSessionOptions(opts = {}) {
  const out = {};

  if (opts.threads !== undefined) {
    if (!Number.isInteger(opts.threads) || opts.threads < 1) {
      throw new TypeError(`threads must be a positive integer, got ${opts.threads}`);
    }
    out.intraOpNumThreads = opts.threads;
    out.interOpNumThreads = 1;
  }

  if (opts.allowSpinning !== undefined) {
    const flag = opts.allowSpinning ? '1' : '0';
    out.extra = {
      session: {
        intra_op: { allow_spinning: flag },
        inter_op: { allow_spinning: flag },
      },
    };
  }

  const raw = opts.sessionOptions || {};
  return {
    ...out,
    ...raw,
    ...(out.extra || raw.extra ? { extra: mergeDeep(out.extra || {}, raw.extra || {}) } : {}),
    executionProviders: opts.executionProviders || raw.executionProviders || ['cpu'],
  };
}

function mergeDeep(a, b) {
  const out = { ...a };
  for (const [k, v] of Object.entries(b)) {
    out[k] = v && typeof v === 'object' && !Array.isArray(v) && typeof out[k] === 'object'
      ? mergeDeep(out[k], v)
      : v;
  }
  return out;
}

module.exports = { buildSessionOptions };
