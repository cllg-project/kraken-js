'use strict';

// ---------------------------------------------------------------------------
// D-FINE detection post-processing.
// All functions are pure and dependency-free, so the browser demo shares them.
// ---------------------------------------------------------------------------

function invertMapping(mapping = {}) {
  const inv = new Map();
  for (const [name, idx] of Object.entries(mapping)) {
    if (!inv.has(idx)) inv.set(idx, name);
  }
  return inv;
}

/**
 * Pick detections from raw model outputs, as dfine-kraken does: top-k over the
 * flattened (query × class) score matrix, then a score threshold. A query can
 * yield several detections if more than one of its classes scores high.
 *
 * @param {Float32Array} scores  (Q × C) sigmoid class scores
 * @param {Float32Array} boxes   (Q × 4) boxes as normalized (cx, cy, w, h)
 * @param {number} Q             number of queries
 * @param {number} C             number of classes
 * @param {object} opts          {topK, scoreThreshold, width, height}
 * @returns {Array<{label: number, score: number, bbox: number[]}>}  by descending score;
 *          `bbox` is `[x0, y0, x1, y1]` in pixels, clamped to the image
 */
function selectDetections(scores, boxes, Q, C, { topK, scoreThreshold, width, height }) {
  // Everything past the top-k is discarded and the threshold is applied after,
  // so thresholding first and then taking the top-k selects the same set.
  const candidates = [];
  for (let i = 0; i < Q * C; i++) {
    if (scores[i] >= scoreThreshold) candidates.push(i);
  }
  candidates.sort((a, b) => scores[b] - scores[a] || a - b);
  candidates.length = Math.min(candidates.length, topK);

  const clampX = v => Math.min(width, Math.max(0, v));
  const clampY = v => Math.min(height, Math.max(0, v));
  return candidates.map((i) => {
    const q = Math.floor(i / C);
    const [cx, cy, w, h] = boxes.subarray(q * 4, q * 4 + 4);
    return {
      label: i % C,
      score: scores[i],
      bbox: [
        clampX((cx - w / 2) * width),
        clampY((cy - h / 2) * height),
        clampX((cx + w / 2) * width),
        clampY((cy + h / 2) * height),
      ],
    };
  });
}

function boxPolygon([x0, y0, x1, y1]) {
  return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
}

function toRegion({ bbox, score }, type) {
  return { bbox, polygon: boxPolygon(bbox), type, score };
}

function toLine({ bbox, score }, type) {
  const [x0, y0, x1, y1] = bbox;
  const polygon = boxPolygon(bbox);
  return {
    bbox,
    polygon,
    obb: {
      cx: (x0 + x1) / 2,
      cy: (y0 + y1) / 2,
      w: x1 - x0,
      h: y1 - y0,
      angle: 0,
      corners: polygon.map(([x, y]) => [Math.round(x), Math.round(y)]),
    },
    type,
    score,
  };
}

module.exports = { selectDetections, invertMapping, toRegion, toLine };
