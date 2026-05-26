'use strict';

// ---------------------------------------------------------------------------
// Heatmap post-processing for segmentation output
// All functions are pure and dependency-free.
// ---------------------------------------------------------------------------

/**
 * Apply sigmoid in-place to a Float32Array.
 */
function sigmoid(arr) {
  for (let i = 0; i < arr.length; i++) {
    arr[i] = 1 / (1 + Math.exp(-arr[i]));
  }
  return arr;
}

/**
 * Element-wise max over specified channels of a (C, H, W) flat array.
 * Returns a Float32Array of length H*W.
 */
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

/**
 * Threshold a (H*W) float array to a binary Uint8Array.
 */
function threshold(channel, H, W, thresh) {
  const mask = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) {
    mask[i] = channel[i] >= thresh ? 1 : 0;
  }
  return mask;
}

/**
 * 4-connected BFS connected components on a binary Uint8Array of shape H×W.
 * Returns { labels: Int32Array (H*W), count: number of components }.
 * Label 0 = background, 1..count = components.
 */
function connectedComponents(mask, H, W) {
  const labels = new Int32Array(H * W);
  let count = 0;
  const queue = [];

  for (let startIdx = 0; startIdx < H * W; startIdx++) {
    if (mask[startIdx] === 0 || labels[startIdx] !== 0) continue;
    count++;
    labels[startIdx] = count;
    queue.length = 0;
    queue.push(startIdx);
    let qi = 0;
    while (qi < queue.length) {
      const idx = queue[qi++];
      const r = (idx / W) | 0;
      const c = idx % W;
      // 4-connected neighbours
      if (r > 0)     { const n = idx - W; if (mask[n] && !labels[n]) { labels[n] = count; queue.push(n); } }
      if (r < H - 1) { const n = idx + W; if (mask[n] && !labels[n]) { labels[n] = count; queue.push(n); } }
      if (c > 0)     { const n = idx - 1; if (mask[n] && !labels[n]) { labels[n] = count; queue.push(n); } }
      if (c < W - 1) { const n = idx + 1; if (mask[n] && !labels[n]) { labels[n] = count; queue.push(n); } }
    }
  }
  return { labels, count };
}

/**
 * Compute a 2×2 symmetric eigen-decomposition.
 * Returns { val: [λ0, λ1], vec: [[vx0,vy0],[vx1,vy1]] } sorted descending by eigenvalue.
 * Used for PCA-based oriented bbox computation.
 */
function eig2x2(a, b, d) {
  // Matrix [[a,b],[b,d]], characteristic polynomial: λ²-(a+d)λ+(ad-b²)=0
  const tr = a + d;
  const det = a * d - b * b;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l0 = tr / 2 + disc;
  const l1 = tr / 2 - disc;

  function eigvec(lambda) {
    // Solve (A - λI)v = 0
    if (Math.abs(b) > 1e-12) {
      const vx = b, vy = lambda - a;
      const norm = Math.sqrt(vx * vx + vy * vy);
      return [vx / norm, vy / norm];
    }
    // Already diagonal
    return lambda >= a ? [1, 0] : [0, 1];
  }

  return { val: [l0, l1], vec: [eigvec(l0), eigvec(l1)] };
}

/**
 * Compute the oriented bounding box of a set of pixel coordinates using PCA.
 *
 * @param {number[]} xs  x-coordinates of pixels
 * @param {number[]} ys  y-coordinates of pixels
 * @returns {{ cx, cy, w, h, angle, corners: number[][] }}
 *   angle: radians of primary axis from +x, in (-π/2, π/2]
 *   corners: [[x,y], ...] clockwise from the corner closest to top-left
 */
function computeOBB(xs, ys) {
  const n = xs.length;
  // Centroid
  let cx = 0, cy = 0;
  for (let i = 0; i < n; i++) { cx += xs[i]; cy += ys[i]; }
  cx /= n; cy /= n;

  // Covariance
  let cxx = 0, cxy = 0, cyy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - cx, dy = ys[i] - cy;
    cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
  }
  cxx /= n; cxy /= n; cyy /= n;

  // Principal axes via 2×2 eigen-decomposition
  const { vec } = eig2x2(cxx, cxy, cyy);
  const [ux, uy] = vec[0]; // primary axis (most variance = text direction)
  const [vx, vy] = vec[1]; // secondary axis (line height)

  // Project all points onto axes
  let uMin = Infinity, uMax = -Infinity, vMin = Infinity, vMax = -Infinity;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - cx, dy = ys[i] - cy;
    const u = dx * ux + dy * uy;
    const v = dx * vx + dy * vy;
    if (u < uMin) uMin = u;
    if (u > uMax) uMax = u;
    if (v < vMin) vMin = v;
    if (v > vMax) vMax = v;
  }

  const w = uMax - uMin;
  const h = vMax - vMin;

  // Adjust center to midpoint of the projected extents
  const uMid = (uMin + uMax) / 2;
  const vMid = (vMin + vMax) / 2;
  const ocx = cx + uMid * ux + vMid * vx;
  const ocy = cy + uMid * uy + vMid * vy;

  // 4 corners in (u, v) space, mapped back to (x, y)
  const corners = [
    [uMin, vMin], [uMax, vMin], [uMax, vMax], [uMin, vMax],
  ].map(([u, v]) => [
    ocx + (u - uMid) * ux + (v - vMid) * vx,
    ocy + (u - uMid) * uy + (v - vMid) * vy,
  ]);

  // Canonical angle: angle of primary axis in (-π/2, π/2]
  let angle = Math.atan2(uy, ux);
  if (angle > Math.PI / 2)  angle -= Math.PI;
  if (angle <= -Math.PI / 2) angle += Math.PI;

  // Sort corners clockwise from top-left (smallest x+y)
  corners.sort((a, b) => (a[0] + a[1]) - (b[0] + b[1]));
  const [tl] = corners.splice(0, 1);
  corners.sort((a, b) => Math.atan2(a[1] - ocy, a[0] - ocx) - Math.atan2(b[1] - ocy, b[0] - ocx));
  const sorted = [tl, ...corners];

  return { cx: ocx, cy: ocy, w, h, angle, corners: sorted };
}

/**
 * For each labeled component (label 1..count), compute its OBB via PCA.
 * Components with fewer than minArea pixels are discarded.
 *
 * @returns {Array<OBB & {label: number, area: number}>}
 */
function extractOrientedBBoxes(labels, count, H, W, minArea = 20) {
  // Collect pixel coords per label
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
    const obb = computeOBB(xs[lbl], ys[lbl]);
    results.push({ ...obb, label: lbl, area });
  }
  return results;
}

/**
 * Detect a vertical column gap from the raw baseline activation profile.
 *
 * For each x-column sums sigmoid(logit) across all rows and all baseline
 * channels. Finds the deepest trough in the central 40 % of the heatmap
 * width. Returns that x if the trough is below `valleyRatio` × the median
 * column sum, otherwise null.
 *
 * Using raw activations (before thresholding) gives a cleaner signal than
 * post-hoc analysis of binarised components: the model itself assigns near-
 * zero probability to the inter-column space regardless of whether adjacent
 * baselines happen to be connected after binarisation.
 *
 * @param {Float32Array} output           Raw model output (C × H × W)
 * @param {number}       H                Heatmap height
 * @param {number}       W                Heatmap width
 * @param {number[]}     baselineChannels  Channel indices for ALL baseline classes
 * @param {object}       [opts]
 * @param {number}       [opts.valleyRatio=0.2]  Trough must be < median × this
 * @returns {number|null}  Heatmap x of the column gap, or null
 */
function findColumnGapFromProfile(output, H, W, baselineChannels, opts = {}) {
  const valleyRatio = opts.valleyRatio ?? 0.7;

  // Sum baseline channel probabilities across all rows for each x-column.
  // Model outputs are already in [0, 1] — do NOT apply sigmoid again.
  const profile = new Float64Array(W);
  for (const c of baselineChannels) {
    const base = c * H * W;
    for (let y = 0; y < H; y++) {
      const row = base + y * W;
      for (let x = 0; x < W; x++) {
        profile[x] += output[row + x];
      }
    }
  }

  const sorted = Float64Array.from(profile).sort();
  const median = sorted[Math.floor(W / 2)];
  if (median === 0) return null;

  const lo = Math.round(W * 0.3);
  const hi = Math.round(W * 0.7);
  let minVal = Infinity, splitX = -1;
  for (let x = lo; x <= hi; x++) {
    if (profile[x] < minVal) { minVal = profile[x]; splitX = x; }
  }

  return (splitX >= 0 && minVal < median * valleyRatio) ? splitX : null;
}

/**
 * Split connected components that straddle a vertical column boundary.
 *
 * For each component that has pixels on both sides of splitX, re-labels the
 * right-side pixels as a new component. The split coordinate should come from
 * `findColumnGapFromProfile` so it reflects the model's own column-gap signal.
 *
 * @param {Int32Array} labels   Component label map (H×W), modified in-place
 * @param {number}     count    Current number of components
 * @param {number}     H
 * @param {number}     W
 * @param {number}     splitX   Heatmap x coordinate of the column gap
 * @returns {number} New component count (>= count)
 */
function splitComponentsAtX(labels, count, H, W, splitX) {
  const hasLeft  = new Uint8Array(count + 1);
  const hasRight = new Uint8Array(count + 1);
  for (let idx = 0; idx < H * W; idx++) {
    const lbl = labels[idx];
    if (lbl === 0) continue;
    if (idx % W <= splitX) hasLeft[lbl]  = 1;
    else                   hasRight[lbl] = 1;
  }

  const remap = new Int32Array(count + 1);
  let newCount = count;
  for (let lbl = 1; lbl <= count; lbl++) {
    if (hasLeft[lbl] && hasRight[lbl]) remap[lbl] = ++newCount;
  }

  for (let idx = 0; idx < H * W; idx++) {
    const lbl = labels[idx];
    if (lbl !== 0 && remap[lbl] && idx % W > splitX) labels[idx] = remap[lbl];
  }
  return newCount;
}

/**
 * Look for a vertical gutter between two page columns.
 *
 * Only activates on landscape images (width > height * 1.2), which is the
 * typical aspect ratio of a double-page spread. Finds the largest cx gap
 * whose midpoint falls in the middle 40% of the image; returns the split
 * x-coordinate if the gap exceeds 5% of image width, otherwise null.
 *
 * @param {Array<{cx: number}>} obbs
 * @param {number} imageWidth
 * @param {number} imageHeight
 * @returns {number|null}
 */
function findColumnSplit(obbs, imageWidth, imageHeight) {
  if (obbs.length < 4) return null;
  if (imageHeight && imageWidth / imageHeight < 1.2) return null;
  const lo = imageWidth * 0.3;
  const hi = imageWidth * 0.7;
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

/**
 * Sort OBBs by reading order.
 *
 * When imageSize is supplied and the image is landscape, looks for a vertical
 * gutter (double-page spread). If found, left and right columns are each sorted
 * by cy independently and concatenated (left first). Pass noColumnSplit:true to
 * disable this detection and always use a plain cy-then-cx sort.
 *
 * For portrait two-column pages, pass `forceSplitX` (in image pixels) derived
 * from `findColumnGapFromProfile` to bypass the landscape gate.
 *
 * @param {Array<OBB>} obbs
 * @param {number}  [imageWidth]
 * @param {number}  [imageHeight]
 * @param {boolean} [noColumnSplit]
 * @param {number}  [forceSplitX]   Pre-computed column split x in image pixels
 * @returns {Array<OBB>}
 */
function sortByReadingOrder(obbs, imageWidth, imageHeight, noColumnSplit, forceSplitX) {
  if (!noColumnSplit) {
    const split = forceSplitX ?? (imageWidth ? findColumnSplit(obbs, imageWidth, imageHeight) : null);
    if (split !== null) {
      const left  = obbs.filter(o => o.cx <= split).sort((a, b) => a.cy - b.cy);
      const right = obbs.filter(o => o.cx >  split).sort((a, b) => a.cy - b.cy);
      return [...left, ...right];
    }
  }
  return [...obbs].sort((a, b) => a.cy !== b.cy ? a.cy - b.cy : a.cx - b.cx);
}

/**
 * Scale OBB coordinates from heatmap space to original image space,
 * and clamp corners to [0, imageW) × [0, imageH).
 */
function scaleOBBs(obbs, scaleX, scaleY, imageW, imageH) {
  return obbs.map(obb => {
    const corners = obb.corners.map(([x, y]) => [
      Math.max(0, Math.min(imageW - 1, Math.round(x * scaleX))),
      Math.max(0, Math.min(imageH - 1, Math.round(y * scaleY))),
    ]);
    return {
      ...obb,
      cx: obb.cx * scaleX,
      cy: obb.cy * scaleY,
      w:  obb.w  * scaleX,
      h:  obb.h  * scaleY,
      corners,
    };
  });
}

module.exports = {
  sigmoid,
  maxChannels,
  threshold,
  connectedComponents,
  extractOrientedBBoxes,
  sortByReadingOrder,
  scaleOBBs,
  findColumnGapFromProfile,
  splitComponentsAtX,
  // exported for testing
  computeOBB,
  eig2x2,
  findColumnSplit,
};
