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

    const crops = await Promise.all(
      lines.map(({ obb }) => extractLineCrop(imageBuffer, obb, imageSize.width, imageSize.height))
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
 * Extract a deskewed line crop from a full-page image buffer.
 * Crops the axis-aligned bbox of the OBB corners (with padding), then rotates.
 */
async function extractLineCrop(imageBuffer, obb, origW, origH) {
  const { h, angle, corners } = obb;

  const xs = corners.map(c => c[0]);
  const ys = corners.map(c => c[1]);
  const pad = Math.ceil(Math.max(h * 0.15, 5));

  const left   = Math.max(0, Math.floor(Math.min(...xs)) - pad);
  const top    = Math.max(0, Math.floor(Math.min(...ys)) - pad);
  const right  = Math.min(origW - 1, Math.ceil(Math.max(...xs)) + pad);
  const bottom = Math.min(origH - 1, Math.ceil(Math.max(...ys)) + pad);

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
