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
 * Sort OBBs by reading order: cy ascending (top to bottom), then cx ascending (left to right).
 */
function sortByReadingOrder(obbs) {
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
  // exported for testing
  computeOBB,
  eig2x2,
};
