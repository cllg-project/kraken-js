'use strict';
const sharp = require('sharp');

/**
 * Preprocess a line image for Kraken recognition inference.
 *
 * Matches Kraken's ImageInputTransforms pipeline (minus CenterNormalizer):
 *   1. Convert to grayscale (channels=1) or RGB (channels=3)
 *   2. Resize to target height, preserving aspect ratio (LANCZOS)
 *   3. Extend left/right by `pad` pixels with white (255)
 *   4. Normalize [0,1] and invert: value = 1.0 - pixel/255
 *
 * @param {string|Buffer} image   Path or raw Buffer
 * @param {object}        meta    {height, channels, pad}
 * @returns {Promise<{data: Float32Array, width: number, height: number}>}
 */
async function preprocessImage(image, meta) {
  const { height, channels, pad } = meta;

  let pipeline = sharp(image, { failOn: 'none' });

  if (channels === 1) {
    pipeline = pipeline.grayscale();
  } else {
    pipeline = pipeline.toColorspace('srgb').removeAlpha();
  }

  // Resize to target height, proportional width
  pipeline = pipeline.resize({ height, fit: 'outside', kernel: 'lanczos3' });

  // Crop to exact height in case resize overshot (shouldn't happen with 'outside', but be safe)
  const resized = await pipeline.raw().toBuffer({ resolveWithObject: true });
  let { data: buf, info } = resized;
  let imgHeight = info.height;
  let imgWidth = info.width;

  // If resize produced a height different from target (e.g., small images), re-resize exactly
  if (imgHeight !== height) {
    const exact = await sharp(buf, { raw: { width: imgWidth, height: imgHeight, channels } })
      .resize({ width: imgWidth, height, fit: 'fill' })
      .raw()
      .toBuffer({ resolveWithObject: true });
    buf = exact.data;
    imgHeight = exact.info.height;
    imgWidth = exact.info.width;
  }

  // Pad left and right with white (255)
  const totalWidth = imgWidth + 2 * pad;
  const nPix = totalWidth * height * channels;
  const padded = Buffer.alloc(nPix, 255);

  // Copy image row by row into padded buffer
  const srcRowBytes = imgWidth * channels;
  const dstRowBytes = totalWidth * channels;
  const leftOffset = pad * channels;
  for (let row = 0; row < height; row++) {
    buf.copy(padded, row * dstRowBytes + leftOffset, row * srcRowBytes, (row + 1) * srcRowBytes);
  }

  // Normalize and invert: 1.0 - pixel/255
  const floatData = new Float32Array(nPix);
  for (let i = 0; i < nPix; i++) {
    floatData[i] = 1.0 - padded[i] / 255.0;
  }

  return { data: floatData, width: totalWidth, height };
}

/**
 * Stack multiple preprocessed images into a batch tensor.
 * Shorter images are zero-padded on the right (safe for CTC — padded frames are blank).
 *
 * @param {Array<{data: Float32Array, width: number, height: number}>} tensors
 * @param {number} channels
 * @returns {{ batchData: Float32Array, batchWidth: number, widths: number[] }}
 */
function buildBatch(tensors, channels) {
  const maxWidth = Math.max(...tensors.map(t => t.width));
  const height = tensors[0].height;
  const N = tensors.length;
  const batchData = new Float32Array(N * channels * height * maxWidth); // zero-filled

  for (let b = 0; b < N; b++) {
    const { data, width } = tensors[b];
    // data layout: [channels * height * width] in CHW order
    for (let c = 0; c < channels; c++) {
      for (let h = 0; h < height; h++) {
        const srcBase = c * height * width + h * width;
        const dstBase = b * channels * height * maxWidth + c * height * maxWidth + h * maxWidth;
        batchData.set(data.subarray(srcBase, srcBase + width), dstBase);
      }
    }
  }

  return { batchData, batchWidth: maxWidth, widths: tensors.map(t => t.width) };
}

/**
 * Convert a preprocessed image {data, width, height} into a CHW Float32Array.
 * sharp returns HWC; this reorders to CHW for ONNX.
 */
function toChw(tensor, channels) {
  const { data, width, height } = tensor;
  if (channels === 1) return data; // HW == CHW when C=1, no reorder needed
  // HWC → CHW
  const out = new Float32Array(channels * height * width);
  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      for (let c = 0; c < channels; c++) {
        out[c * height * width + h * width + w] = data[(h * width + w) * channels + c];
      }
    }
  }
  return out;
}

/**
 * Preprocess a full-page image for Kraken segmentation inference.
 *
 * @param {string|Buffer} image   Path or raw Buffer
 * @param {object}        meta    {height, channels, one_channel_mode}
 * @returns {Promise<{data: Float32Array, width: number, height: number, actualChannels: number}>}
 */
async function preprocessPageImage(image, meta) {
  const { height, channels } = meta;

  let pipeline = sharp(image, { failOn: 'none' });
  if (channels === 1) {
    pipeline = pipeline.grayscale();
  } else {
    pipeline = pipeline.toColorspace('srgb').removeAlpha();
  }
  pipeline = pipeline.resize({ height, fit: 'outside', kernel: 'lanczos3' });

  const { data: buf, info } = await pipeline.raw().toBuffer({ resolveWithObject: true });
  const nPix = info.width * info.height * info.channels;
  const floatData = new Float32Array(nPix);
  for (let i = 0; i < nPix; i++) {
    floatData[i] = buf[i] / 255.0;
  }

  return { data: floatData, width: info.width, height: info.height, actualChannels: info.channels };
}

module.exports = { preprocessImage, preprocessPageImage, buildBatch, toChw };
