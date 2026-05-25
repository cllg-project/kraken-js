'use strict';

/**
 * Build a label→character lookup from the codec.
 * codec: { char: [label_id, ...] }  (multi-label entries = ligatures)
 * Returns Map<string, string> where key is label IDs joined by ',' e.g. "4,5" → "æ"
 */
function buildL2C(codec) {
  const l2c = new Map();
  for (const [char, ids] of Object.entries(codec)) {
    l2c.set(ids.join(','), char);
  }
  return l2c;
}

/**
 * Greedy CTC decoder.
 * Applies softmax internally so confidences are in [0,1] even when raw logits are passed.
 *
 * @param {Float32Array} logits  Raw logits or softmax probs, layout [C * W] (C classes, W timesteps)
 * @param {number}       C       Number of classes (including blank at 0)
 * @param {number}       W       Number of timesteps
 * @returns {Array<{label: number, t0: number, t1: number, conf: number}>}
 */
function greedyCTC(logits, C, W) {
  // Apply softmax per timestep so confidences are true probabilities
  const probs = new Float32Array(logits.length);
  for (let t = 0; t < W; t++) {
    let maxVal = -Infinity;
    for (let c = 0; c < C; c++) maxVal = Math.max(maxVal, logits[c * W + t]);
    let sum = 0;
    for (let c = 0; c < C; c++) {
      const v = Math.exp(logits[c * W + t] - maxVal);
      probs[c * W + t] = v;
      sum += v;
    }
    for (let c = 0; c < C; c++) probs[c * W + t] /= sum;
  }
  const results = [];
  let prevLabel = 0;
  let runStart = 0;
  let runMaxConf = 0;
  let runLabel = 0;

  // Accumulate runs, then emit on change
  for (let t = 0; t < W; t++) {
    // argmax over classes at timestep t
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let c = 0; c < C; c++) {
      const v = probs[c * W + t];
      if (v > maxVal) { maxVal = v; maxIdx = c; }
    }

    if (maxIdx !== prevLabel) {
      // emit previous run if it was not blank
      if (prevLabel !== 0) {
        results.push({ label: prevLabel, t0: runStart, t1: t - 1, conf: runMaxConf });
      }
      runStart = t;
      runMaxConf = maxVal;
      runLabel = maxIdx;
    } else {
      if (maxVal > runMaxConf) runMaxConf = maxVal;
    }
    prevLabel = maxIdx;
  }
  // flush last run
  if (prevLabel !== 0) {
    results.push({ label: prevLabel, t0: runStart, t1: W - 1, conf: runMaxConf });
  }
  return results;
}

/**
 * Decode a CTC label sequence to characters using the codec.
 * Handles multi-label (ligature) codes via greedy longest-prefix scan.
 *
 * @param {Array<{label, t0, t1, conf}>} ctcLabels  Output of greedyCTC
 * @param {Map<string, string>}          l2c         Output of buildL2C
 * @returns {Array<{char: string, t0: number, t1: number, conf: number}>}
 */
function decodeCodec(ctcLabels, l2c) {
  const chars = [];
  let i = 0;
  while (i < ctcLabels.length) {
    // try longest match first (up to 4 labels, covers all realistic ligatures)
    let matched = false;
    for (let len = Math.min(4, ctcLabels.length - i); len >= 1; len--) {
      const key = ctcLabels.slice(i, i + len).map(x => x.label).join(',');
      if (l2c.has(key)) {
        const group = ctcLabels.slice(i, i + len);
        const conf = group.reduce((s, x) => s + x.conf, 0) / group.length;
        chars.push({
          char: l2c.get(key),
          t0: group[0].t0,
          t1: group[len - 1].t1,
          conf,
        });
        i += len;
        matched = true;
        break;
      }
    }
    if (!matched) {
      // unknown label — skip
      i++;
    }
  }
  return chars;
}

module.exports = { buildL2C, greedyCTC, decodeCodec };
