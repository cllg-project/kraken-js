/**
 * Browser-side kraken-js pipeline.
 *
 * Uses:
 *   - onnxruntime-web (ESM) for inference
 *   - JSZip (global, loaded via script tag) for .js_mlmodel unpacking
 *   - Canvas API for image preprocessing (replaces sharp)
 *
 * The segmentation + recognition logic mirrors src/segmenter.js,
 * src/recognizer.js, src/preprocess.js, src/heatmap.js and src/decode.js.
 */

// ort is loaded as a UMD global via <script> tag in index.html.
// Set wasmPaths so the runtime can fetch its .wasm workers from the same CDN.
/* global ort */
// Point to the same CDN so the .wasm worker files are fetched from there
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

// ---------------------------------------------------------------------------
// .js_mlmodel loader (uses JSZip global)
// ---------------------------------------------------------------------------

async function loadJsMlmodel(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status} ${resp.statusText}`);
  const buf = await resp.arrayBuffer();
  const zip = await JSZip.loadAsync(buf);

  const metaStr  = await zip.file('metadata.json').async('string');
  const onnxBuf  = await zip.file('model.onnx').async('arraybuffer');
  const meta     = JSON.parse(metaStr);

  const session  = await ort.InferenceSession.create(onnxBuf, {
    executionProviders: ['wasm'],
  });

  return { session, meta };
}

// ---------------------------------------------------------------------------
// Image preprocessing — Canvas API (mirrors src/preprocess.js)
// ---------------------------------------------------------------------------

/**
 * Draw an HTMLImageElement (or ImageBitmap) onto a canvas scaled to targetH,
 * preserving aspect ratio.
 */
function resizeToHeight(img, targetH) {
  const origW = img.naturalWidth  ?? img.width;
  const origH = img.naturalHeight ?? img.height;
  const scale = targetH / origH;
  const targetW = Math.max(1, Math.round(origW * scale));

  const c = document.createElement('canvas');
  c.width  = targetW;
  c.height = targetH;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, targetW, targetH);
  return c;
}

/**
 * Preprocess a page image for segmentation.
 *
 * Steps (mirrors preprocessPageImage in src/preprocess.js):
 *   1. Resize to model height (proportional width)
 *   2. Read pixels (RGB)
 *   3. Normalize + invert: 1 - pixel/255
 *   4. Reorder HWC → CHW
 *
 * Returns { data: Float32Array (CHW), width, height }
 */
function preprocessPageCanvas(img, meta) {
  const { height: targetH, channels } = meta;
  const c   = resizeToHeight(img, targetH);
  const ctx = c.getContext('2d');
  const { width: W, height: H } = c;

  const raw = ctx.getImageData(0, 0, W, H).data; // RGBA Uint8ClampedArray

  if (channels === 1) {
    // Grayscale: average R,G,B → broadcast to 1 channel, normalize + invert
    const out = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) {
      const r = raw[i * 4], g = raw[i * 4 + 1], b = raw[i * 4 + 2];
      const gray = (r + g + b) / 3;
      out[i] = 1.0 - gray / 255.0;
    }
    // CHW with C=1: layout is just H*W
    return { data: out, width: W, height: H };
  }

  // RGB: normalize + invert, then HWC → CHW
  const hwc = new Float32Array(H * W * 3);
  for (let i = 0; i < H * W; i++) {
    hwc[i * 3    ] = 1.0 - raw[i * 4    ] / 255.0;
    hwc[i * 3 + 1] = 1.0 - raw[i * 4 + 1] / 255.0;
    hwc[i * 3 + 2] = 1.0 - raw[i * 4 + 2] / 255.0;
  }

  // HWC → CHW
  const chw = new Float32Array(3 * H * W);
  for (let h = 0; h < H; h++) {
    for (let w = 0; w < W; w++) {
      for (let ch = 0; ch < 3; ch++) {
        chw[ch * H * W + h * W + w] = hwc[(h * W + w) * 3 + ch];
      }
    }
  }

  return { data: chw, width: W, height: H };
}

/**
 * Preprocess a line crop (canvas or image element) for recognition.
 *
 * Mirrors preprocessImage in src/preprocess.js.
 * Returns Float32Array in CHW layout.
 */
function preprocessLineCanvas(imgEl, meta) {
  const { height: targetH, channels, pad } = meta;
  const c   = resizeToHeight(imgEl, targetH);
  const W0  = c.width;
  const ctx = c.getContext('2d');
  const raw = ctx.getImageData(0, 0, W0, targetH).data;

  const totalW = W0 + 2 * pad;
  const nPix   = totalW * targetH * channels;

  // Build padded HWC buffer (white = 255)
  const padded = new Float32Array(nPix);
  for (let i = 0; i < nPix; i++) padded[i] = 1.0; // pre-invert: white=0 → 1-0/255=1

  if (channels === 1) {
    for (let row = 0; row < targetH; row++) {
      for (let col = 0; col < W0; col++) {
        const r = raw[(row * W0 + col) * 4];
        const g = raw[(row * W0 + col) * 4 + 1];
        const b = raw[(row * W0 + col) * 4 + 2];
        const gray = (r + g + b) / 3;
        padded[row * totalW + (pad + col)] = 1.0 - gray / 255.0;
      }
    }
    // CHW with C=1 is same as H*W
    return { data: padded, width: totalW, height: targetH };
  }

  // RGB
  const paddedRgb = new Uint8Array(totalW * targetH * 3).fill(255);
  for (let row = 0; row < targetH; row++) {
    for (let col = 0; col < W0; col++) {
      const dst = (row * totalW + pad + col) * 3;
      const src = (row * W0 + col) * 4;
      paddedRgb[dst    ] = raw[src    ];
      paddedRgb[dst + 1] = raw[src + 1];
      paddedRgb[dst + 2] = raw[src + 2];
    }
  }

  const nPixFull = totalW * targetH * 3;
  const floatData = new Float32Array(nPixFull);
  for (let i = 0; i < nPixFull; i++) floatData[i] = 1.0 - paddedRgb[i] / 255.0;

  // HWC → CHW
  const chw = new Float32Array(nPixFull);
  for (let h = 0; h < targetH; h++) {
    for (let w = 0; w < totalW; w++) {
      for (let ch = 0; ch < channels; ch++) {
        chw[ch * targetH * totalW + h * totalW + w] = floatData[(h * totalW + w) * channels + ch];
      }
    }
  }
  return { data: chw, width: totalW, height: targetH };
}

// ---------------------------------------------------------------------------
// Heatmap post-processing (mirrors src/heatmap.js — pure functions)
// ---------------------------------------------------------------------------

function maxChannels(data, C, H, W, channelIndices) {
  const out = new Float32Array(H * W).fill(-Infinity);
  for (const c of channelIndices) {
    const base = c * H * W;
    for (let i = 0; i < H * W; i++) {
      if (data[base + i] > out[i]) out[i] = data[base + i];
    }
  }
  return out;
}

function threshold(channel, thresh) {
  const mask = new Uint8Array(channel.length);
  for (let i = 0; i < channel.length; i++) mask[i] = channel[i] >= thresh ? 1 : 0;
  return mask;
}

function connectedComponents(mask, H, W) {
  const labels = new Int32Array(H * W);
  let count = 0;
  const queue = [];
  for (let startIdx = 0; startIdx < H * W; startIdx++) {
    if (!mask[startIdx] || labels[startIdx]) continue;
    count++;
    labels[startIdx] = count;
    queue.length = 0;
    queue.push(startIdx);
    let qi = 0;
    while (qi < queue.length) {
      const idx = queue[qi++];
      const r = (idx / W) | 0, c = idx % W;
      if (r > 0)     { const n = idx - W; if (mask[n] && !labels[n]) { labels[n] = count; queue.push(n); } }
      if (r < H - 1) { const n = idx + W; if (mask[n] && !labels[n]) { labels[n] = count; queue.push(n); } }
      if (c > 0)     { const n = idx - 1; if (mask[n] && !labels[n]) { labels[n] = count; queue.push(n); } }
      if (c < W - 1) { const n = idx + 1; if (mask[n] && !labels[n]) { labels[n] = count; queue.push(n); } }
    }
  }
  return { labels, count };
}

function eig2x2(a, b, d) {
  const tr = a + d, det = a * d - b * b;
  const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l0 = tr / 2 + disc, l1 = tr / 2 - disc;
  function eigvec(lambda) {
    if (Math.abs(b) > 1e-12) {
      const vx = b, vy = lambda - a, norm = Math.sqrt(vx * vx + vy * vy);
      return [vx / norm, vy / norm];
    }
    return lambda >= a ? [1, 0] : [0, 1];
  }
  return { val: [l0, l1], vec: [eigvec(l0), eigvec(l1)] };
}

function computeOBB(xs, ys) {
  const n = xs.length;
  let cx = 0, cy = 0;
  for (let i = 0; i < n; i++) { cx += xs[i]; cy += ys[i]; }
  cx /= n; cy /= n;

  let cxx = 0, cxy = 0, cyy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - cx, dy = ys[i] - cy;
    cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
  }
  cxx /= n; cxy /= n; cyy /= n;

  const { vec } = eig2x2(cxx, cxy, cyy);
  const [ux, uy] = vec[0], [vx, vy] = vec[1];

  let uMin = Infinity, uMax = -Infinity, vMin = Infinity, vMax = -Infinity;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - cx, dy = ys[i] - cy;
    const u = dx * ux + dy * uy, v = dx * vx + dy * vy;
    if (u < uMin) uMin = u; if (u > uMax) uMax = u;
    if (v < vMin) vMin = v; if (v > vMax) vMax = v;
  }

  const w = uMax - uMin, h = vMax - vMin;
  const uMid = (uMin + uMax) / 2, vMid = (vMin + vMax) / 2;
  const ocx = cx + uMid * ux + vMid * vx;
  const ocy = cy + uMid * uy + vMid * vy;

  const corners = [[uMin, vMin], [uMax, vMin], [uMax, vMax], [uMin, vMax]]
    .map(([u, v]) => [ocx + (u - uMid) * ux + (v - vMid) * vx,
                      ocy + (u - uMid) * uy + (v - vMid) * vy]);

  let angle = Math.atan2(uy, ux);
  if (angle > Math.PI / 2)   angle -= Math.PI;
  if (angle <= -Math.PI / 2) angle += Math.PI;

  corners.sort((a, b) => (a[0] + a[1]) - (b[0] + b[1]));
  const [tl] = corners.splice(0, 1);
  corners.sort((a, b) => Math.atan2(a[1] - ocy, a[0] - ocx) - Math.atan2(b[1] - ocy, b[0] - ocx));
  const sorted = [tl, ...corners];

  return { cx: ocx, cy: ocy, w, h, angle, corners: sorted };
}

function extractOrientedBBoxes(labels, count, H, W, minArea = 20) {
  const xs = Array.from({ length: count + 1 }, () => []);
  const ys = Array.from({ length: count + 1 }, () => []);
  for (let idx = 0; idx < H * W; idx++) {
    const lbl = labels[idx];
    if (lbl === 0) continue;
    xs[lbl].push(idx % W);
    ys[lbl].push((idx / W) | 0);
  }
  const results = [];
  for (let lbl = 1; lbl <= count; lbl++) {
    const area = xs[lbl].length;
    if (area < minArea) continue;
    results.push({ ...computeOBB(xs[lbl], ys[lbl]), label: lbl, area });
  }
  return results;
}

function findColumnSplit(obbs, imageWidth, imageHeight) {
  if (obbs.length < 4) return null;
  if (imageHeight && imageWidth / imageHeight < 1.2) return null;
  const lo = imageWidth * 0.3, hi = imageWidth * 0.7;
  const cxs = obbs.map(o => o.cx).sort((a, b) => a - b);
  let maxGap = 0, splitX = null;
  for (let i = 1; i < cxs.length; i++) {
    const mid = (cxs[i] + cxs[i - 1]) / 2;
    if (mid < lo || mid > hi) continue;
    const gap = cxs[i] - cxs[i - 1];
    if (gap > maxGap) { maxGap = gap; splitX = mid; }
  }
  return (splitX !== null && maxGap > imageWidth * 0.05) ? splitX : null;
}

function sortByReadingOrder(obbs, imageWidth, imageHeight, noColumnSplit) {
  if (!noColumnSplit && imageWidth) {
    const split = findColumnSplit(obbs, imageWidth, imageHeight);
    if (split !== null) {
      const left  = obbs.filter(o => o.cx <= split).sort((a, b) => a.cy - b.cy);
      const right = obbs.filter(o => o.cx >  split).sort((a, b) => a.cy - b.cy);
      return [...left, ...right];
    }
  }
  return [...obbs].sort((a, b) => a.cy !== b.cy ? a.cy - b.cy : a.cx - b.cx);
}

function scaleOBBs(obbs, scaleX, scaleY, imageW, imageH) {
  return obbs.map(obb => ({
    ...obb,
    cx: obb.cx * scaleX,
    cy: obb.cy * scaleY,
    w:  obb.w  * scaleX,
    h:  obb.h  * scaleY,
    corners: obb.corners.map(([x, y]) => [
      Math.max(0, Math.min(imageW - 1, Math.round(x * scaleX))),
      Math.max(0, Math.min(imageH - 1, Math.round(y * scaleY))),
    ]),
  }));
}

// ---------------------------------------------------------------------------
// CTC decode (mirrors src/decode.js — pure functions)
// ---------------------------------------------------------------------------

function buildL2C(codec) {
  const l2c = new Map();
  for (const [char, ids] of Object.entries(codec)) l2c.set(ids.join(','), char);
  return l2c;
}

function greedyCTC(logits, C, W) {
  const probs = new Float32Array(logits.length);
  for (let t = 0; t < W; t++) {
    let maxVal = -Infinity;
    for (let c = 0; c < C; c++) maxVal = Math.max(maxVal, logits[c * W + t]);
    let sum = 0;
    for (let c = 0; c < C; c++) {
      const v = Math.exp(logits[c * W + t] - maxVal);
      probs[c * W + t] = v; sum += v;
    }
    for (let c = 0; c < C; c++) probs[c * W + t] /= sum;
  }
  const results = [];
  let prevLabel = 0, runStart = 0, runMaxConf = 0;
  for (let t = 0; t < W; t++) {
    let maxVal = -Infinity, maxIdx = 0;
    for (let c = 0; c < C; c++) {
      const v = probs[c * W + t];
      if (v > maxVal) { maxVal = v; maxIdx = c; }
    }
    if (maxIdx !== prevLabel) {
      if (prevLabel !== 0) results.push({ label: prevLabel, t0: runStart, t1: t - 1, conf: runMaxConf });
      runStart = t; runMaxConf = maxVal;
    } else {
      if (maxVal > runMaxConf) runMaxConf = maxVal;
    }
    prevLabel = maxIdx;
  }
  if (prevLabel !== 0) results.push({ label: prevLabel, t0: runStart, t1: W - 1, conf: runMaxConf });
  return results;
}

function decodeCodec(ctcLabels, l2c) {
  const chars = [];
  let i = 0;
  while (i < ctcLabels.length) {
    let matched = false;
    for (let len = Math.min(4, ctcLabels.length - i); len >= 1; len--) {
      const key = ctcLabels.slice(i, i + len).map(x => x.label).join(',');
      if (l2c.has(key)) {
        const group = ctcLabels.slice(i, i + len);
        const conf  = group.reduce((s, x) => s + x.conf, 0) / group.length;
        chars.push({ char: l2c.get(key), t0: group[0].t0, t1: group[len - 1].t1, conf });
        i += len; matched = true; break;
      }
    }
    if (!matched) i++;
  }
  return chars;
}

// ---------------------------------------------------------------------------
// BrowserSegmenter
// ---------------------------------------------------------------------------

class BrowserSegmenter {
  constructor(session, meta, opts = {}) {
    this._session       = session;
    this._meta          = meta;
    this._noColumnSplit = opts.noColumnSplit || false;
  }

  static async create(url, opts = {}) {
    const { session, meta } = await loadJsMlmodel(url);
    return new BrowserSegmenter(session, meta, opts);
  }

  async segment(imgEl) {
    const origW = imgEl.naturalWidth  ?? imgEl.width;
    const origH = imgEl.naturalHeight ?? imgEl.height;

    const { data: chw, width: W, height: H } = preprocessPageCanvas(imgEl, this._meta);

    const tensor = new ort.Tensor('float32', chw, [1, this._meta.channels, H, W]);
    const output = await this._session.run({ input: tensor });
    const outKey = Object.keys(output)[0];
    const out    = output[outKey];

    const [, C, Hout, Wout] = out.dims;
    const outData = out.data;

    // baseline class indices: all entries under class_mapping.baselines
    const baselineIndices = Object.values(this._meta.class_mapping.baselines);

    const merged  = maxChannels(outData, C, Hout, Wout, baselineIndices);
    const mask    = threshold(merged, 0.5);
    const { labels, count } = connectedComponents(mask, Hout, Wout);
    let obbs = extractOrientedBBoxes(labels, count, Hout, Wout, 20);

    const scaleX = origW / Wout;
    const scaleY = origH / Hout;
    obbs = scaleOBBs(obbs, scaleX, scaleY, origW, origH);

    // Determine type per OBB by which class_mapping.baselines value has highest mean
    const classMap = this._meta.class_mapping.baselines;
    const classEntries = Object.entries(classMap); // [[name, idx], ...]

    const lines = sortByReadingOrder(obbs, origW, origH, this._noColumnSplit)
      .map(obb => {
        // find dominant baseline class for this component
        let bestType = classEntries[0][0];
        // simple heuristic: lowest class index if only one channel, else DefaultLine
        if (classEntries.length > 1) {
          const dominantIdx = baselineIndices[0];
          bestType = classEntries.find(([, idx]) => idx === dominantIdx)?.[0] ?? classEntries[0][0];
        }
        return { obb, type: bestType };
      });

    return { lines, imageSize: { width: origW, height: origH } };
  }
}

// ---------------------------------------------------------------------------
// BrowserRecognizer
// ---------------------------------------------------------------------------

class BrowserRecognizer {
  constructor(session, meta) {
    this._session = session;
    this._meta    = meta;
    this._l2c     = buildL2C(meta.codec);
  }

  static async create(url) {
    const { session, meta } = await loadJsMlmodel(url);
    return new BrowserRecognizer(session, meta);
  }

  async recognize(imgEl) {
    const { data: chw, width: W, height: H } = preprocessLineCanvas(imgEl, this._meta);

    const tensor = new ort.Tensor('float32', chw, [1, this._meta.channels, H, W]);
    const output = await this._session.run({ input: tensor });
    const outKey = Object.keys(output)[0];
    const out    = output[outKey];

    // output dims: [N, C, W_out] — data is in (C, W) layout for batch 0
    const [, C, Wout] = out.dims;
    const logits     = out.data.subarray(0, C * Wout); // (C, W) layout

    const ctcLabels = greedyCTC(logits, C, Wout);
    const chars     = decodeCodec(ctcLabels, this._l2c);

    const inputWidth = W;
    const scaledChars = chars.map(ch => ({
      char: ch.char,
      conf: ch.conf,
      x0: Math.round(ch.t0 * inputWidth / Wout),
      x1: Math.round(ch.t1 * inputWidth / Wout),
    }));

    return { text: scaledChars.map(c => c.char).join(''), chars: scaledChars };
  }
}

// ---------------------------------------------------------------------------
// Line crop extraction (mirrors pipeline.js extractLineCrop using Canvas)
// ---------------------------------------------------------------------------

function estimateLineHeight(lines, imageSize) {
  if (lines.length < 2) return Math.round(imageSize.height / 30);
  const cys  = lines.map(l => l.obb.cy).slice().sort((a, b) => a - b);
  const gaps = [];
  for (let i = 1; i < cys.length; i++) {
    const g = cys[i] - cys[i - 1];
    if (g > 0) gaps.push(g);
  }
  if (gaps.length === 0) return Math.round(imageSize.height / 30);
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)];
}

function extractLineCropCanvas(sourceCanvas, obb, origW, origH, lineHeight, topline) {
  const { cy, angle, corners } = obb;
  const expandUp   = topline ? lineHeight * 0.25 : lineHeight * 0.85;
  const expandDown = topline ? lineHeight * 0.85 : lineHeight * 0.25;

  const xs   = corners.map(c => c[0]);
  const hPad = Math.ceil(lineHeight * 0.1);

  const left   = Math.max(0, Math.floor(Math.min(...xs)) - hPad);
  const top    = Math.max(0, Math.floor(cy - expandUp));
  const right  = Math.min(origW - 1, Math.ceil(Math.max(...xs)) + hPad);
  const bottom = Math.min(origH - 1, Math.ceil(cy + expandDown));

  const cropW = Math.max(1, right - left);
  const cropH = Math.max(1, bottom - top);
  // bounds exposed for overlay drawing
  extractLineCropCanvas._lastBounds = { left, top, right, bottom };

  const c   = document.createElement('canvas');
  const ctx = c.getContext('2d');

  const angleDeg = angle * 180 / Math.PI;
  if (Math.abs(angleDeg) >= 0.5) {
    // Rotate the crop around its centre
    const rad = -angleDeg * Math.PI / 180;
    const cos = Math.abs(Math.cos(rad)), sin = Math.abs(Math.sin(rad));
    const rotW = Math.ceil(cropW * cos + cropH * sin);
    const rotH = Math.ceil(cropW * sin + cropH * cos);
    c.width  = rotW;
    c.height = rotH;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, rotW, rotH);
    ctx.save();
    ctx.translate(rotW / 2, rotH / 2);
    ctx.rotate(rad);
    ctx.drawImage(sourceCanvas, left, top, cropW, cropH, -cropW / 2, -cropH / 2, cropW, cropH);
    ctx.restore();
  } else {
    c.width  = cropW;
    c.height = cropH;
    ctx.drawImage(sourceCanvas, left, top, cropW, cropH, 0, 0, cropW, cropH);
  }

  return c;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Run the full segmentation + recognition pipeline on an image element.
 *
 * @param {HTMLImageElement} imgEl
 * @param {string}           segUrl   URL of segmentation .js_mlmodel
 * @param {string}           recUrl   URL of recognition .js_mlmodel
 * @param {{ onStatus?: (msg:string)=>void, onLine?: (line:object)=>void }} opts
 */
export async function runPipeline(imgEl, segUrl, recUrl, opts = {}) {
  const { onStatus = () => {}, onLine = () => {}, noColumnSplit = false } = opts;

  onStatus('Loading segmentation model…');
  const segmenter = await BrowserSegmenter.create(segUrl, { noColumnSplit });

  onStatus('Loading recognition model…');
  const recognizer = await BrowserRecognizer.create(recUrl);

  onStatus('Segmenting page…');
  const { lines, imageSize } = await segmenter.segment(imgEl);

  if (lines.length === 0) {
    onStatus('No lines detected.');
    return [];
  }

  // Build a canvas from the source image for crop extraction
  const sourceCanvas = document.createElement('canvas');
  sourceCanvas.width  = imageSize.width;
  sourceCanvas.height = imageSize.height;
  sourceCanvas.getContext('2d').drawImage(imgEl, 0, 0);

  const lineHeight = estimateLineHeight(lines, imageSize);
  const topline    = segmenter._meta.topline || false;

  const yield_ = () => new Promise(r => setTimeout(r, 0));

  const results = [];
  for (let i = 0; i < lines.length; i++) {
    const { obb, type } = lines[i];
    onStatus(`Recognizing line ${i + 1} / ${lines.length}…`);
    await yield_(); // release the main thread so the browser can repaint

    const cropCanvas = extractLineCropCanvas(
      sourceCanvas, obb,
      imageSize.width, imageSize.height,
      lineHeight, topline
    );
    const cropBounds = { ...extractLineCropCanvas._lastBounds };

    const { text, chars } = await recognizer.recognize(cropCanvas);
    const result = { obb, type, text, chars, cropBounds };
    results.push(result);
    onLine(result);
  }

  return results;
}
