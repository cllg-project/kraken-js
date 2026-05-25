'use strict';
const fs = require('node:fs');
const sharp = require('sharp');
const { KrakenSegmenter } = require('./segmenter');
const { KrakenRecognizer } = require('./recognizer');

class KrakenPipeline {
  constructor(segmenter, recognizer, opts) {
    this._segmenter = segmenter;
    this._recognizer = recognizer;
    this._opts = opts;
  }

  static async create(segmenterPath, recognizerPath, opts = {}) {
    const [segmenter, recognizer] = await Promise.all([
      KrakenSegmenter.create(segmenterPath, opts.segmenter || {}),
      KrakenRecognizer.create(recognizerPath, opts.recognizer || {}),
    ]);
    return new KrakenPipeline(segmenter, recognizer, opts);
  }

  async process(image) {
    const imageBuffer = typeof image === 'string' ? fs.readFileSync(image) : image;

    const { lines, imageSize } = await this._segmenter.segment(imageBuffer);
    if (lines.length === 0) return [];

    // The model predicts thin baselines (~1–2px in heatmap space), so obb.h is
    // meaningless as a line height. Estimate from median inter-baseline spacing.
    const lineHeight = estimateLineHeight(lines, imageSize);
    const topline = this._segmenter._meta.topline || false;

    const crops = await Promise.all(
      lines.map(({ obb }) =>
        extractLineCrop(imageBuffer, obb, imageSize.width, imageSize.height, lineHeight, topline)
      )
    );

    const recognized = await Promise.all(crops.map(crop => this._recognizer.recognize(crop)));

    return lines.map(({ obb, type }, i) => ({
      obb,
      type,
      text: recognized[i].text,
      chars: recognized[i].chars,
    }));
  }
}

/**
 * Estimate line height from the median gap between consecutive baseline cy values.
 * Falls back to imageHeight/30 when there are too few lines.
 */
function estimateLineHeight(lines, imageSize) {
  if (lines.length < 2) return Math.round(imageSize.height / 30);
  const cys = lines.map(l => l.obb.cy).slice().sort((a, b) => a - b);
  const gaps = [];
  for (let i = 1; i < cys.length; i++) {
    const g = cys[i] - cys[i - 1];
    if (g > 0) gaps.push(g);
  }
  if (gaps.length === 0) return Math.round(imageSize.height / 30);
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)];
}

/**
 * Extract a deskewed line crop from a full-page image buffer.
 *
 * Uses estimated line height (not obb.h, which is just the baseline width) to
 * determine the vertical crop extent. For topline=false (Kraken default), text
 * sits above the baseline → expand mostly upward.
 */
async function extractLineCrop(imageBuffer, obb, origW, origH, lineHeight, topline) {
  const { cx, cy, angle, corners } = obb;

  const expandUp   = topline ? lineHeight * 0.25 : lineHeight * 0.85;
  const expandDown = topline ? lineHeight * 0.85 : lineHeight * 0.25;

  const xs  = corners.map(c => c[0]);
  const hPad = Math.ceil(lineHeight * 0.1);

  const left   = Math.max(0, Math.floor(Math.min(...xs)) - hPad);
  const top    = Math.max(0, Math.floor(cy - expandUp));
  const right  = Math.min(origW - 1, Math.ceil(Math.max(...xs)) + hPad);
  const bottom = Math.min(origH - 1, Math.ceil(cy + expandDown));

  const cropW = Math.max(1, right - left);
  const cropH = Math.max(1, bottom - top);

  const angleDeg = angle * 180 / Math.PI;
  let pipeline = sharp(imageBuffer).extract({ left, top, width: cropW, height: cropH });

  if (Math.abs(angleDeg) >= 0.5) {
    pipeline = pipeline.rotate(-angleDeg, { background: { r: 255, g: 255, b: 255, alpha: 1 } });
  }

  return pipeline.toBuffer();
}

module.exports = { KrakenPipeline };
