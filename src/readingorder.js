'use strict';
const ort = require('onnxruntime-node');
const { loadJsMlmodel } = require('./loader');
const { buildSessionOptions } = require('./session');

const EPS = 1.1920929e-7; // torch.finfo(torch.float).eps

/**
 * Neural reading order (Kraken's ROMLP): a small MLP scores every ordered pair of
 * lines ("does A come before B?"); a greedy decoder turns the pair matrix into a
 * sequence. Feature construction and decoding mirror
 * `kraken.lib.segmentation.neural_reading_order`, so the order matches Kraken's.
 */
class ReadingOrder {
  constructor(session, meta) {
    this.session = session;
    this._meta = meta;
    this._classMapping = meta.class_mapping || {};
    this._numClasses = Math.max(0, ...Object.values(this._classMapping)) + 1;
  }

  /**
   * @param {string} modelPath  `.js_mlmodel` of a reading-order model
   * @param {object} [opts]     `threads`, `allowSpinning`, `sessionOptions`, `executionProviders`
   */
  static async create(modelPath, opts = {}) {
    const { onnxBytes, metadata } = loadJsMlmodel(modelPath);
    if (metadata.model_type !== 'reading_order') {
      throw new Error(`${modelPath}: not a reading-order model (model_type=${metadata.model_type})`);
    }
    const session = await ort.InferenceSession.create(onnxBytes, buildSessionOptions(opts));
    return new ReadingOrder(session, metadata);
  }

  /**
   * Features of one line: one-hot class, then centre, start and end of its baseline,
   * all divided by the image size.
   * @param {{obb?: object, bbox?: number[], type?: string}} line
   */
  _features(line, width, height) {
    const f = new Array(this._numClasses).fill(0);
    f[this._classMapping[line.type] ?? 0] = 1;
    let cx, cy, sx, sy, ex, ey;
    if (line.obb) {
      // obb.w runs along the baseline; angle in (-π/2, π/2], so start is the left end.
      const { cx: x, cy: y, w, angle } = line.obb;
      const dx = Math.cos(angle) * w / 2, dy = Math.sin(angle) * w / 2;
      cx = x; cy = y; sx = x - dx; sy = y - dy; ex = x + dx; ey = y + dy;
    } else {
      const [x0, y0, x1, y1] = line.bbox;
      cx = (x0 + x1) / 2; cy = (y0 + y1) / 2; sx = x0; sy = cy; ex = x1; ey = cy;
    }
    f.push(cx / width, cy / height, sx / width, sy / height, ex / width, ey / height);
    return f;
  }

  /**
   * @param {Array} lines  segmenter lines (with `obb` or `bbox`, and `type`)
   * @param {{width: number, height: number}} imageSize
   * @returns {Promise<number[]>} indices of `lines` in reading order
   */
  async order(lines, imageSize) {
    const n = lines.length;
    if (n <= 1) return n ? [0] : [];
    const feats = lines.map(l => this._features(l, imageSize.width, imageSize.height));
    const size = this._meta.feature_size;
    const pairs = n * (n - 1);
    const input = new Float32Array(pairs * size);
    let p = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        input.set(feats[i], p * size);
        input.set(feats[j], p * size + feats[i].length);
        p++;
      }
    }
    const out = await this.session.run({ input: new ort.Tensor('float32', input, [pairs, size]) });
    const scores = out.output.data;
    const P = Array.from({ length: n }, () => new Float64Array(n));
    p = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) P[i][j] = scores[p++];
    return greedyOrderDecoder(P);
  }
}

/** Port of kraken's `_greedy_order_decoder` (float32 eps, same tie-breaking: first max). */
function greedyOrderDecoder(P) {
  const N = P.length;
  const A = P.map(r => Float64Array.from(r, v => Math.fround(v + EPS)));
  const sym = A.map((r, i) => Float64Array.from(r, (_, j) => (A[i][j] + (1 - A[j][i])) / 2));
  for (let i = 0; i < N; i++) sym[i][i] = EPS;
  const lP = sym.map(r => Float64Array.from(r, v => Math.log(v)));
  for (let i = 0; i < N; i++) lP[i][i] = 0;
  const path = [];
  const used = new Set();
  for (let t = 0; t < N; t++) {
    for (let i = 0; i < N; i++) {
      let best = -Infinity, idx = 0;
      for (let r = 0; r < N; r++) {
        let s = 0;
        for (let c = 0; c < N; c++) s += lP[r][c];
        if (s > best) { best = s; idx = r; }
      }
      if (!used.has(idx)) {
        used.add(idx);
        path.push(idx);
        for (let c = 0; c < N; c++) lP[idx][c] = lP[c][idx];
        for (let r = 0; r < N; r++) lP[r][idx] = 0;
        break;
      }
    }
  }
  return path;
}

module.exports = { ReadingOrder, greedyOrderDecoder };
