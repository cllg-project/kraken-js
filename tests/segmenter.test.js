'use strict';
const { test, describe, before } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const { KrakenSegmenter } = require('../src/segmenter');

const MODEL    = path.join(__dirname, 'fixtures/segmentation.js_mlmodel');
const FULLPAGE = path.join(__dirname, 'fixtures/fullpage.png');
const ALTO_XML = path.join(__dirname, 'fixtures/rscir_0035-2217_1984_num_58_1_2999.pdf_page_7.xml');

/** Parse ALTO baselines into [{cx, cy, content}] sorted by cy. */
function parseAlto(xmlPath) {
  const xml = fs.readFileSync(xmlPath, 'utf8');
  const lines = [];
  for (const m of xml.matchAll(/<TextLine[^>]+BASELINE="([^"]+)"[\s\S]*?CONTENT="([^"]*)"/g)) {
    const pts   = m[1].trim().split(/\s+/).map(Number);
    const xs    = pts.filter((_, i) => i % 2 === 0);
    const ys    = pts.filter((_, i) => i % 2 === 1);
    const xSpan = Math.max(...xs) - Math.min(...xs);
    lines.push({
      cx: xs.reduce((a, b) => a + b, 0) / xs.length,
      cy: ys.reduce((a, b) => a + b, 0) / ys.length,
      xSpan,
      content: m[2],
    });
  }
  return lines.sort((a, b) => a.cy - b.cy);
}

let segmenter;
let result;

before(async () => {
  segmenter = await KrakenSegmenter.create(MODEL);
  result = await segmenter.segment(FULLPAGE);
});

// ---------------------------------------------------------------------------
// KrakenSegmenter.create
// ---------------------------------------------------------------------------

describe('KrakenSegmenter.create', () => {
  test('returns a KrakenSegmenter instance', () => {
    assert.ok(segmenter instanceof KrakenSegmenter);
  });

  test('metadata has required keys', () => {
    const m = segmenter._meta;
    assert.ok('model_type' in m);
    assert.ok('height' in m);
    assert.ok('channels' in m);
    assert.ok('class_mapping' in m);
  });

  test('metadata height is 1800', () => {
    assert.equal(segmenter._meta.height, 1800);
  });

  test('metadata channels is 3', () => {
    assert.equal(segmenter._meta.channels, 3);
  });

  test('class_mapping has baselines group', () => {
    assert.ok('baselines' in segmenter._meta.class_mapping);
  });

  test('class_mapping baselines has DefaultLine', () => {
    assert.ok('DefaultLine' in segmenter._meta.class_mapping.baselines);
  });

  test('throws on non-existent model file', async () => {
    await assert.rejects(() => KrakenSegmenter.create('/no/such/model.js_mlmodel'));
  });
});

// ---------------------------------------------------------------------------
// segment — result shape
// ---------------------------------------------------------------------------

describe('segment result shape', () => {
  test('returns object with lines and imageSize', () => {
    assert.ok('lines' in result);
    assert.ok('imageSize' in result);
  });

  test('imageSize matches fullpage.png dimensions', () => {
    assert.equal(result.imageSize.width, 2479);
    assert.equal(result.imageSize.height, 3508);
  });

  test('lines is a non-empty array', () => {
    assert.ok(Array.isArray(result.lines));
    assert.ok(result.lines.length > 0, `expected lines, got ${result.lines.length}`);
  });

  test('plausible number of lines for a manuscript page (10–150)', () => {
    assert.ok(result.lines.length >= 10, `too few lines: ${result.lines.length}`);
    assert.ok(result.lines.length <= 150, `too many lines: ${result.lines.length}`);
  });
});

// ---------------------------------------------------------------------------
// segment — line structure
// ---------------------------------------------------------------------------

describe('segment line structure', () => {
  test('each line has obb and type', () => {
    for (const line of result.lines) {
      assert.ok('obb' in line, 'missing obb');
      assert.ok('type' in line, 'missing type');
    }
  });

  test('each obb has required fields', () => {
    for (const { obb } of result.lines) {
      for (const key of ['cx', 'cy', 'w', 'h', 'angle', 'corners']) {
        assert.ok(key in obb, `missing obb.${key}`);
      }
    }
  });

  test('each obb has 4 corners', () => {
    for (const { obb } of result.lines) {
      assert.equal(obb.corners.length, 4);
    }
  });

  test('each corner is [x, y] within image bounds', () => {
    const { width, height } = result.imageSize;
    for (const { obb } of result.lines) {
      for (const [x, y] of obb.corners) {
        assert.ok(x >= 0 && x < width,  `corner x=${x} out of [0, ${width})`);
        assert.ok(y >= 0 && y < height, `corner y=${y} out of [0, ${height})`);
      }
    }
  });

  test('angle is in (-π/2, π/2]', () => {
    for (const { obb } of result.lines) {
      assert.ok(obb.angle > -Math.PI / 2, `angle too small: ${obb.angle}`);
      assert.ok(obb.angle <= Math.PI / 2,  `angle too large: ${obb.angle}`);
    }
  });

  test('type is DefaultLine or DefaultLine-Margin', () => {
    const valid = new Set(['DefaultLine', 'DefaultLine-Margin']);
    for (const { type } of result.lines) {
      assert.ok(valid.has(type), `unexpected type: ${type}`);
    }
  });
});

// ---------------------------------------------------------------------------
// segment — reading order
// ---------------------------------------------------------------------------

describe('segment reading order', () => {
  test('lines are sorted by cy ascending', () => {
    for (let i = 1; i < result.lines.length; i++) {
      assert.ok(
        result.lines[i].obb.cy >= result.lines[i - 1].obb.cy,
        `line ${i} cy=${result.lines[i].obb.cy} < previous cy=${result.lines[i - 1].obb.cy}`
      );
    }
  });
});

// ---------------------------------------------------------------------------
// segment — ALTO ground-truth comparison
//
// The ALTO file is the reference segmentation produced by Python Kraken.
// Each ALTO line should be matched by at least one pipeline detection within
// CY_TOL pixels, and the matched detection's cx should be within CX_TOL pixels.
// Short/margin lines (width < 200px in ALTO) are excluded from the cx check.
// ---------------------------------------------------------------------------

const CY_TOL = 40;   // px — vertical tolerance for baseline matching
const CX_TOL = 200;  // px — horizontal centre tolerance

describe('segment ALTO ground-truth comparison', () => {
  let altoLines;

  before(() => { altoLines = parseAlto(ALTO_XML); });

  test('ALTO has expected number of lines (39)', () => {
    assert.equal(altoLines.length, 39);
  });

  test('every ALTO line is matched by a detection within ' + CY_TOL + 'px vertically', () => {
    const detectedCys = result.lines.map(l => l.obb.cy);
    const missed = altoLines.filter(al =>
      !detectedCys.some(cy => Math.abs(cy - al.cy) <= CY_TOL)
    );
    assert.equal(missed.length, 0,
      'Unmatched ALTO lines:\n' + missed.map(l => `  cy=${Math.round(l.cy)} "${l.content}"`).join('\n')
    );
  });

  test('no spurious detections (every detected line matches an ALTO line within ' + CY_TOL + 'px)', () => {
    const altoCys = altoLines.map(l => l.cy);
    const spurious = result.lines.filter(pl =>
      !altoCys.some(cy => Math.abs(cy - pl.obb.cy) <= CY_TOL)
    );
    assert.equal(spurious.length, 0,
      'Spurious detections:\n' + spurious.map(l => `  cy=${Math.round(l.obb.cy)}`).join('\n')
    );
  });

  test('matched wide lines have cx within ' + CX_TOL + 'px of ALTO baseline centre', () => {
    // Only check lines whose baseline spans > 200px (skips margin numbers, short titles etc.)
    // Match each wide ALTO line against the closest wide detection to avoid pairing
    // body text with nearby narrow margin-number detections at the same cy.
    const wideAltoLines = altoLines.filter(al => al.xSpan > 200);
    const failures = [];
    for (const al of wideAltoLines) {
      const match = result.lines
        .filter(pl => Math.abs(pl.obb.cy - al.cy) <= CY_TOL && pl.obb.w > 200)
        .sort((a, b) => Math.abs(a.obb.cy - al.cy) - Math.abs(b.obb.cy - al.cy))[0];
      if (!match) continue;
      const diff = Math.abs(match.obb.cx - al.cx);
      if (diff > CX_TOL) {
        failures.push(`cy=${Math.round(al.cy)} cx_alto=${Math.round(al.cx)} cx_det=${Math.round(match.obb.cx)} Δ=${Math.round(diff)} "${al.content.slice(0, 40)}"`);
      }
    }
    assert.equal(failures.length, 0, 'cx mismatch:\n' + failures.map(s => '  ' + s).join('\n'));
  });
});
