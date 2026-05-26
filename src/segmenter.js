'use strict';
const ort = require('onnxruntime-node');
const { loadJsMlmodel } = require('./loader');
const { preprocessPageImage, toChw } = require('./preprocess');
const {
  maxChannels, threshold, connectedComponents,
  extractOrientedBBoxes, scaleOBBs, sortByReadingOrder,
} = require('./heatmap');

/**
 * Locates text lines on a full page image and returns oriented bounding boxes (OBBs).
 *
 * Each OBB is computed via PCA on the connected-component pixels in the baseline heatmap.
 * Lines are returned in reading order; landscape double-page spreads are detected
 * automatically and split into left/right columns (disable with `noColumnSplit`).
 */
class KrakenSegmenter {
  constructor(session, meta, opts) {
    this.session = session;
    this._meta = meta;
    this._opts = opts;
  }

  /**
   * Load a `.js_mlmodel` segmentation model.
   *
   * @param {string} modelPath  Path to the `.js_mlmodel` file
   * @param {object} [opts]
   * @param {number}   [opts.threshold=0.5]       Sigmoid threshold for baseline heatmap binarisation
   * @param {number}   [opts.minArea=20]           Minimum connected-component area (heatmap pixels)
   * @param {boolean}  [opts.noColumnSplit=false]  Disable double-page column detection
   * @param {string[]} [opts.executionProviders=['cpu']]  ONNX Runtime execution providers
   * @returns {Promise<KrakenSegmenter>}
   */
  static async create(modelPath, opts = {}) {
    const { onnxBytes, metadata } = await loadJsMlmodel(modelPath);
    const ep = opts.executionProviders || ['cpu'];
    const session = await ort.InferenceSession.create(onnxBytes, { executionProviders: ep });
    return new KrakenSegmenter(session, metadata, opts);
  }

  /**
   * Segment a page image into text lines.
   *
   * @param {string|Buffer} image  File path or raw image Buffer
   * @returns {Promise<{
   *   lines: Array<{ obb: {cx,cy,w,h,angle,corners}, type: string }>,
   *   imageSize: { width: number, height: number }
   * }>}
   *
   * `obb` fields (all in original image pixels):
   *   - `cx`, `cy`   — baseline centre
   *   - `w`          — length along the text direction
   *   - `h`          — baseline thickness (~1–2 px; not the full glyph height)
   *   - `angle`      — radians from +x axis, in (-π/2, π/2]
   *   - `corners`    — `[[x,y]×4]` clockwise from top-left
   *
   * `type` is one of the class names from `metadata.class_mapping.baselines`
   * (e.g. `'DefaultLine'` or `'DefaultLine-Margin'`).
   */
  async segment(image) {
    const sharp = require('sharp');

    const imgInfo = await sharp(image, { failOn: 'none' }).metadata();
    const origW = imgInfo.width;
    const origH = imgInfo.height;

    const preprocessed = await preprocessPageImage(image, this._meta);
    const { data, width: modelW, height: modelH, actualChannels } = preprocessed;
    const chw = toChw({ data, width: modelW, height: modelH }, actualChannels);

    const tensor = new ort.Tensor('float32', chw, [1, actualChannels, modelH, modelW]);
    const results = await this.session.run({ input: tensor });
    const output = results.output.data;
    const C = results.output.dims[1];
    const H_out = results.output.dims[2];
    const W_out = results.output.dims[3];

    const classMapping = this._meta.class_mapping || {};
    const baselineEntries = classMapping.baselines || {};
    const baselineChannels = Object.values(baselineEntries);
    const channelToClass = {};
    for (const [name, idx] of Object.entries(baselineEntries)) {
      channelToClass[idx] = name;
    }

    if (baselineChannels.length === 0) {
      return { lines: [], imageSize: { width: origW, height: origH } };
    }

    const baseline = maxChannels(output, C, H_out, W_out, baselineChannels);
    const thresh = this._opts.threshold ?? 0.5;
    const mask = threshold(baseline, H_out, W_out, thresh);
    const { labels, count } = connectedComponents(mask, H_out, W_out);
    const rawObbs = extractOrientedBBoxes(labels, count, H_out, W_out, this._opts.minArea ?? 20);

    // One-pass accumulation of per-component, per-channel activation sums
    const compSums = Array.from({ length: count + 1 }, () => ({}));
    for (let idx = 0; idx < H_out * W_out; idx++) {
      const lbl = labels[idx];
      if (lbl === 0) continue;
      for (const c of baselineChannels) {
        compSums[lbl][c] = (compSums[lbl][c] || 0) + output[c * H_out * W_out + idx];
      }
    }

    const scaleX = origW / W_out;
    const scaleY = origH / H_out;
    const scaledObbs = scaleOBBs(rawObbs, scaleX, scaleY, origW, origH);

    const lines = scaledObbs.map((obb, i) => {
      const sums = compSums[rawObbs[i].label] || {};
      let bestChannel = baselineChannels[0];
      let bestSum = -Infinity;
      for (const c of baselineChannels) {
        if ((sums[c] || 0) > bestSum) { bestSum = sums[c] || 0; bestChannel = c; }
      }
      return { obb, type: channelToClass[bestChannel] || 'DefaultLine' };
    });

    const tagged   = lines.map((l, i) => ({ ...l.obb, _i: i }));
    const ordered  = sortByReadingOrder(tagged, origW, origH, this._opts.noColumnSplit);
    const sortedLines = ordered.map(o => lines[o._i]);
    lines.length = 0;
    lines.push(...sortedLines);

    return { lines, imageSize: { width: origW, height: origH } };
  }
}

module.exports = { KrakenSegmenter };
