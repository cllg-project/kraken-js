'use strict';
const fs = require('node:fs');
const sharp = require('sharp');
const { KrakenSegmenter } = require('./segmenter');
const { KrakenRecognizer } = require('./recognizer');

/**
 * End-to-end OCR pipeline: segment a page image, deskew line crops, and recognize text.
 */
class KrakenPipeline {
  constructor(segmenter, recognizer, opts) {
    this._segmenter = segmenter;
    this._recognizer = recognizer;
    this._opts = opts;
  }

  /**
   * Create a pipeline from two `.js_mlmodel` files.
   *
   * @param {string} segmenterPath   Path to the segmentation model
   * @param {string} recognizerPath  Path to the recognition model
   * @param {object} [opts]
   * @param {number}   [opts.expandUp=0.85]    Fraction of estimated line height to include above baseline
   * @param {number}   [opts.expandDown=0.35]  Fraction of estimated line height to include below baseline
   * @param {object}   [opts.segmenter={}]     Options forwarded to {@link KrakenSegmenter.create}
   * @param {object}   [opts.recognizer={}]    Options forwarded to {@link KrakenRecognizer.create}
   * @returns {Promise<KrakenPipeline>}
   */
  static async create(segmenterPath, recognizerPath, opts = {}) {
    const [segmenter, recognizer] = await Promise.all([
      KrakenSegmenter.create(segmenterPath, opts.segmenter || {}),
      KrakenRecognizer.create(recognizerPath, opts.recognizer || {}),
    ]);
    return new KrakenPipeline(segmenter, recognizer, opts);
  }

  /**
   * Process a full page image end-to-end.
   *
   * @param {string|Buffer} image  File path or raw image Buffer
   * @returns {Promise<Array<{
   *   obb:   { cx, cy, w, h, angle, corners },
   *   type:  string,
   *   text:  string,
   *   chars: Array<{ char: string, conf: number, x0: number, x1: number }>
   * }>>}
   *
   * Results are in reading order. `chars` `x0`/`x1` are relative to the line crop,
   * not the full page image.
   */
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
        extractLineCrop(imageBuffer, obb, imageSize.width, imageSize.height, lineHeight, topline, this._opts)
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
  // Use lines in their existing order (columns already grouped by sortByReadingOrder).
  // A global cy sort would interleave two columns, producing near-zero gaps between
  // lines at the same vertical position on opposite pages — biasing the median to ~0.
  // Filter: skip gaps < 2px (cross-column noise) and > 30% of image height (the jump
  // from the bottom of one column back to the top of the next).
  const maxGap = imageSize.height * 0.3;
  const gaps = [];
  for (let i = 1; i < lines.length; i++) {
    const g = lines[i].obb.cy - lines[i - 1].obb.cy;
    if (g > 2 && g < maxGap) gaps.push(g);
  }
  if (gaps.length === 0) return Math.round(imageSize.height / 30);
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)];
}

/**
 * Extract a deskewed line crop from a full-page image buffer.
 *
 * Strategy: orient the crop along the OBB angle, then straighten it.
 *   1. Compute the 4 corners of the desired crop in image space (OBB expanded
 *      perpendicular to the baseline by expandUp/expandDown).
 *   2. Extract the axis-aligned bounding box of those 4 corners.
 *   3. Rotate the extracted patch by -angle so text becomes horizontal.
 *   4. Re-extract the now-axis-aligned crop from the rotated patch.
 */
async function extractLineCrop(imageBuffer, obb, origW, origH, lineHeight, topline, opts = {}) {
  const { cx, cy, angle, w: obbW } = obb;

  const upRatio    = opts.expandUp   ?? (topline ? 0.35 : 0.85);
  const downRatio  = opts.expandDown ?? (topline ? 0.85 : 0.35);
  const expandUp   = lineHeight * upRatio;
  const expandDown = lineHeight * downRatio;
  const hPad = Math.ceil(lineHeight * 0.1);
  const hw   = obbW / 2 + hPad;
  const angleDeg = angle * 180 / Math.PI;

  // Unit vectors: along text direction and perpendicular "above baseline"
  const cosA = Math.cos(angle), sinA = Math.sin(angle);
  const vx = sinA, vy = -cosA; // 90° CCW from primary axis = "above" in screen coords

  // 4 corners of the oriented crop region (TL, TR, BR, BL)
  const rc = [
    [cx - hw * cosA + expandUp   * vx, cy - hw * sinA + expandUp   * vy],
    [cx + hw * cosA + expandUp   * vx, cy + hw * sinA + expandUp   * vy],
    [cx + hw * cosA - expandDown * vx, cy + hw * sinA - expandDown * vy],
    [cx - hw * cosA - expandDown * vx, cy - hw * sinA - expandDown * vy],
  ];

  // Axis-aligned bounding box of those corners → pre-extract region
  const rxs = rc.map(c => c[0]), rys = rc.map(c => c[1]);
  const preLeft   = Math.max(0, Math.floor(Math.min(...rxs)));
  const preTop    = Math.max(0, Math.floor(Math.min(...rys)));
  const preRight  = Math.min(origW - 1, Math.ceil(Math.max(...rxs)));
  const preBottom = Math.min(origH - 1, Math.ceil(Math.max(...rys)));
  const preW = Math.max(1, preRight - preLeft);
  const preH = Math.max(1, preBottom - preTop);

  if (Math.abs(angleDeg) < 0.5) {
    return sharp(imageBuffer)
      .extract({ left: preLeft, top: preTop, width: preW, height: preH })
      .toBuffer();
  }

  // Dimensions of the rotated patch (sharp expands canvas to avoid clipping)
  const absRad = Math.abs(angle);
  const rotW = Math.ceil(preW * Math.cos(absRad) + preH * Math.sin(absRad));
  const rotH = Math.ceil(preH * Math.cos(absRad) + preW * Math.sin(absRad));

  // Where does the line centre (cx, cy) land in the rotated patch?
  const rad  = -angle;
  const cosR = Math.cos(rad), sinR = Math.sin(rad);
  const dx   = (cx - preLeft) - preW / 2;
  const dy   = (cy - preTop)  - preH / 2;
  const newCx = rotW / 2 + dx * cosR - dy * sinR;
  const newCy = rotH / 2 + dx * sinR + dy * cosR;

  // Final crop centred on the now-horizontal baseline
  const fL = Math.max(0,       Math.round(newCx - hw));
  const fR = Math.min(rotW - 1, Math.round(newCx + hw));
  const fT = Math.max(0,       Math.round(newCy - expandUp));
  const fB = Math.min(rotH - 1, Math.round(newCy + expandDown));

  return sharp(imageBuffer)
    .extract({ left: preLeft, top: preTop, width: preW, height: preH })
    .rotate(-angleDeg, { background: { r: 255, g: 255, b: 255, alpha: 1 } })
    .extract({ left: fL, top: fT, width: Math.max(1, fR - fL), height: Math.max(1, fB - fT) })
    .toBuffer();
}

module.exports = { KrakenPipeline };
