'use strict';
const { KrakenRecognizer } = require('./recognizer');
const { spikes, merge, candidates } = require('./merge');

const ROLES = ['base', 'new', 'spec'];

/**
 * Three recognisers read the same line crop; signs enter the base text only where `new` and `spec`
 * agree on them (see ./merge.js). Same `recognize()` contract as {@link KrakenRecognizer}, plus
 * `edits`, `candidates`, `lowconf` and the per-model readings.
 */
class MultiRecognizer {
  /**
   * @param {{base: KrakenRecognizer, new: KrakenRecognizer, spec: KrakenRecognizer}} models
   * @param {object} [mergeParams]  overrides of DEFAULT_PARAMS in ./merge.js
   */
  constructor(models, mergeParams = {}) {
    for (const r of ROLES) if (!models[r]) throw new TypeError(`MultiRecognizer needs a "${r}" model`);
    this._m = models;
    this._params = mergeParams;
  }

  /**
   * @param {{base: string, new: string, spec: string}} paths  `.js_mlmodel` files
   * @param {object} [opts]  forwarded to {@link KrakenRecognizer.create} for each model; `merge` holds merge params.
   *                         `threads`, when set, is the TOTAL budget and is split across the three sessions.
   */
  static async create(paths, opts = {}) {
    for (const r of ROLES) if (!paths[r]) throw new TypeError(`MultiRecognizer needs a "${r}" model path`);
    const { merge: mergeParams, ...rest } = opts;
    if (Number.isInteger(rest.threads) && rest.threads > 0) rest.threads = Math.max(1, Math.floor(rest.threads / 3));
    const made = await Promise.all(ROLES.map(r => KrakenRecognizer.create(paths[r], rest)));
    return new MultiRecognizer(Object.fromEntries(ROLES.map((r, i) => [r, made[i]])), mergeParams);
  }

  /** Spikes of one model, x rescaled to the base model's pixel scale (models may differ in input height). */
  _spikes(role, chars) {
    const k = this._m.base._meta.height / this._m[role]._meta.height;
    return spikes(chars).map(s => ({ ...s, x: s.x * k }));
  }

  async recognize(image) {
    const [b, n, s] = await Promise.all(ROLES.map(r => this._m[r].recognize(image)));
    const bs = this._spikes('base', b.chars);
    const ns = this._spikes('new', n.chars);
    const ss = this._spikes('spec', s.chars);
    const { text, edits, flags } = merge(bs, ns, ss, this._params);
    return {
      text,
      chars: b.chars,
      edits,
      candidates: candidates(bs, ns, edits, this._params),
      lowconf: flags,
      models: { base: b.text, new: n.text, spec: s.text },
    };
  }
}

module.exports = { MultiRecognizer };
