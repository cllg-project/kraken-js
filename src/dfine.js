'use strict';
const ort = require('onnxruntime-node');
const { loadJsMlmodel } = require('./loader');
const { buildSessionOptions } = require('./session');
const { preprocessDetectionImage } = require('./preprocess');
const { sortByReadingOrder } = require('./heatmap');
const { selectDetections, invertMapping, toRegion, toLine } = require('./detections');

/**
 * Detects text lines and layout regions as axis-aligned boxes with a D-FINE model
 * (exported from dfine-kraken). All model variants — nano, small, medium, large and
 * extra_large — share the same graph signature and are handled identically.
 *
 * Post-processing mirrors dfine-kraken's `DFINEModel.predict`: the top
 * `num_top_queries` (query, class) pairs by score are kept, those under the score
 * threshold dropped, and each surviving detection becomes a line or a region
 * according to the model's class mapping. Like kraken, no NMS is applied, so a
 * model may report overlapping boxes for the same object.
 */
class DFineSegmenter {
  constructor(session, meta, opts = {}) {
    this.session = session;
    this._meta = meta;
    this._opts = opts;

    // Invert class_mapping; several names may share an index → keep the first.
    this._lineMap = invertMapping(meta.class_mapping?.lines);
    this._regionMap = invertMapping(meta.class_mapping?.regions);
  }

  /**
   * Load a `.js_mlmodel` D-FINE model.
   *
   * @param {string} modelPath  Path to the `.js_mlmodel` file
   * @param {object} [opts]
   * @param {number}   [opts.scoreThreshold=0.5]  Minimum class score for a detection
   * @param {boolean}  [opts.noColumnSplit=false] Disable double-page column detection
   *                                              when ordering lines
   * @param {string[]} [opts.executionProviders=['cpu']]  ONNX Runtime execution providers
   * @param {number}   [opts.threads]         Threads per inference (default: one per physical core)
   * @param {boolean}  [opts.allowSpinning]   `false` stops idle threads from busy-waiting
   * @param {object}   [opts.sessionOptions]  Raw ONNX Runtime session options
   * @returns {Promise<DFineSegmenter>}
   */
  static async create(modelPath, opts = {}) {
    const { onnxBytes, metadata } = await loadJsMlmodel(modelPath);
    return DFineSegmenter.fromModel(onnxBytes, metadata, opts);
  }

  /** Build a segmenter from an already-loaded ONNX graph and its metadata. */
  static async fromModel(onnxBytes, metadata, opts = {}) {
    if (metadata.architecture !== 'dfine') {
      throw new Error(`not a D-FINE model (architecture: ${metadata.architecture})`);
    }
    const session = await ort.InferenceSession.create(onnxBytes, buildSessionOptions(opts));
    return new DFineSegmenter(session, metadata, opts);
  }

  /**
   * Detect lines and regions on a page image.
   *
   * @param {string|Buffer} image  File path or raw image Buffer
   * @returns {Promise<{
   *   lines:   Array<{ bbox, polygon, obb, type, score, regions: number[] }>,
   *   regions: Array<{ bbox, polygon, type, score }>,
   *   imageSize: { width: number, height: number }
   * }>}
   *
   * All coordinates are in original image pixels:
   *   - `bbox`    — `[x0, y0, x1, y1]`, clamped to the image
   *   - `polygon` — the box as `[[x,y]×4]` clockwise from top-left
   *   - `obb`     — the box in the OBB shape the VGSL segmenter returns
   *                 (`angle` 0, `h` the full line height), so D-FINE lines can go
   *                 anywhere a VGSL line can
   *   - `regions` — on lines: indices into `regions` of the regions whose box
   *                 contains the line's centre
   *
   * Lines are in reading order (top-to-bottom, with the same double-page column
   * split as `KrakenSegmenter`); regions are ordered top-to-bottom, left-to-right.
   */
  async segment(image) {
    const sharp = require('sharp');
    const { width: origW, height: origH } = await sharp(image, { failOn: 'none' }).metadata();

    const { data, width, height } = await preprocessDetectionImage(image, this._meta);
    const input = new ort.Tensor('float32', data, [1, 3, height, width]);
    const { scores, boxes } = await this.session.run({ input });

    const detections = selectDetections(scores.data, boxes.data, scores.dims[1], scores.dims[2], {
      topK: this._meta.num_top_queries ?? 300,
      scoreThreshold: this._opts.scoreThreshold ?? 0.5,
      width: origW,
      height: origH,
    });

    const regions = [];
    const lines = [];
    for (const det of detections) {
      if (this._regionMap.has(det.label)) {
        regions.push(toRegion(det, this._regionMap.get(det.label)));
      }
      if (this._lineMap.has(det.label)) {
        lines.push(toLine(det, this._lineMap.get(det.label)));
      }
    }

    regions.sort((a, b) => a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0]);
    for (const line of lines) {
      const { cx, cy } = line.obb;
      line.regions = [];
      regions.forEach(({ bbox: [x0, y0, x1, y1] }, i) => {
        if (cx > x0 && cx < x1 && cy > y0 && cy < y1) line.regions.push(i);
      });
    }

    const tagged = lines.map((l, i) => ({ ...l.obb, _i: i }));
    const ordered = sortByReadingOrder(tagged, origW, origH, this._opts.noColumnSplit)
      .map(o => lines[o._i]);

    return { lines: ordered, regions, imageSize: { width: origW, height: origH } };
  }
}

module.exports = { DFineSegmenter, selectDetections };
