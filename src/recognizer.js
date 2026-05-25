'use strict';
const ort = require('onnxruntime-node');
const { loadJsMlmodel } = require('./loader');
const { preprocessImage, buildBatch, toChw } = require('./preprocess');
const { buildL2C, greedyCTC, decodeCodec } = require('./decode');

/**
 * Kraken OCR line recognizer backed by an ONNX model.
 *
 * Supports any model exported from a Kraken .mlmodel or .safetensors via
 * export_kraken_onnx.py — adapts automatically to the model's height,
 * channel count, padding, and codec.
 *
 * Usage:
 *   const r = await KrakenRecognizer.create('./model.js_mlmodel');
 *   const { text } = await r.recognize('./line.png');
 *   const results  = await r.recognizeBatch(['./a.png', './b.png']);
 */
class KrakenRecognizer {
  /**
   * @param {ort.InferenceSession} session
   * @param {object}               metadata  {height, channels, pad, codec, vgsl, …}
   */
  constructor(session, metadata) {
    this._session  = session;
    this._meta     = metadata;
    this._l2c      = buildL2C(metadata.codec);
  }

  /**
   * Create a KrakenRecognizer from a .js_mlmodel file.
   *
   * @param {string}   modelPath
   * @param {object}   [opts]
   * @param {string[]} [opts.executionProviders]  e.g. ['cuda','cpu'], ['coreml','cpu']
   *                   Defaults to ['cpu']. Pass ['webgpu','cpu'] in browser contexts.
   * @returns {Promise<KrakenRecognizer>}
   */
  static async create(modelPath, opts = {}) {
    const { onnxBytes, metadata } = loadJsMlmodel(modelPath);
    const providers = opts.executionProviders ?? ['cpu'];
    const session = await ort.InferenceSession.create(onnxBytes, {
      executionProviders: providers,
    });
    return new KrakenRecognizer(session, metadata);
  }

  /**
   * Recognize a single line image.
   *
   * @param {string|Buffer} image  File path or raw image Buffer
   * @returns {Promise<{text: string, chars: Array<{char,conf,x0,x1}>}>}
   */
  async recognize(image) {
    const results = await this.recognizeBatch([image]);
    return results[0];
  }

  /**
   * Recognize a batch of line images in a single forward pass.
   * Images are padded on the right to the widest in the batch.
   *
   * @param {Array<string|Buffer>} images
   * @returns {Promise<Array<{text: string, chars: Array<{char,conf,x0,x1}>}>>}
   */
  async recognizeBatch(images) {
    const { channels } = this._meta;

    // Preprocess all images concurrently
    const rawTensors = await Promise.all(
      images.map(img => preprocessImage(img, this._meta))
    );

    // Reorder HWC→CHW for each
    const tensors = rawTensors.map(t => ({
      data:   toChw(t, channels),
      width:  t.width,
      height: t.height,
    }));

    const N = tensors.length;
    const height = tensors[0].height;
    const { batchData, batchWidth, widths } = buildBatch(tensors, channels);

    const inputTensor = new ort.Tensor('float32', batchData, [N, channels, height, batchWidth]);
    const feeds = { input: inputTensor };
    const results = await this._session.run(feeds);

    // output: (N, C, W_out) — Float32Array
    const output     = results['output'];
    const [, C, Wout] = output.dims;
    const outData    = output.data; // Float32Array of length N*C*Wout

    return tensors.map((t, b) => {
      // Each image's output width is proportional to its input width
      const outW = Math.round((t.width / batchWidth) * Wout);
      // Slice this image's output from the batch: shape [C, Wout], but we only read outW cols
      const imgProbs = outData.subarray(b * C * Wout, (b + 1) * C * Wout);

      const ctcLabels = greedyCTC(imgProbs, C, outW);
      const charSeq   = decodeCodec(ctcLabels, this._l2c);

      // Scale t0/t1 from output-space to input-pixel-space (excluding padding)
      const { pad } = this._meta;
      const inputImgWidth = t.width - 2 * pad; // width without padding
      const scale = inputImgWidth / outW;

      const chars = charSeq.map(({ char, t0, t1, conf }) => ({
        char,
        conf: Math.round(conf * 1000) / 1000,
        x0:   Math.max(0, Math.round(t0 * scale)),
        x1:   Math.round(t1 * scale),
      }));

      return { text: chars.map(c => c.char).join(''), chars };
    });
  }
}

module.exports = { KrakenRecognizer };
