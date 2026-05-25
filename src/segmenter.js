'use strict';
const ort = require('onnxruntime-node');
const { loadJsMlmodel } = require('./loader');
const { preprocessPageImage, toChw } = require('./preprocess');
const {
  maxChannels, threshold, connectedComponents,
  extractOrientedBBoxes, scaleOBBs,
} = require('./heatmap');

class KrakenSegmenter {
  constructor(session, meta, opts) {
    this.session = session;
    this._meta = meta;
    this._opts = opts;
  }

  static async create(modelPath, opts = {}) {
    const { onnxBytes, metadata } = await loadJsMlmodel(modelPath);
    const ep = opts.executionProviders || ['cpu'];
    const session = await ort.InferenceSession.create(onnxBytes, { executionProviders: ep });
    return new KrakenSegmenter(session, metadata, opts);
  }

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

    // Sort by reading order: cy ascending, then cx ascending
    lines.sort((a, b) => a.obb.cy !== b.obb.cy ? a.obb.cy - b.obb.cy : a.obb.cx - b.obb.cx);

    return { lines, imageSize: { width: origW, height: origH } };
  }
}

module.exports = { KrakenSegmenter };
