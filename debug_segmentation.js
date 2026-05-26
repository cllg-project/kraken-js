#!/usr/bin/env node
'use strict';
/**
 * Debug tool: run segmentation on an image and render OBBs + crop regions as a PNG.
 *
 * Usage:
 *   node debug_segmentation.js <image> [seg_model] [output]
 *
 * Defaults:
 *   seg_model  tests/fixtures/segmentation.js_mlmodel
 *   output     debug_segmentation.png
 *
 * The output shows:
 *   Red polygon      — raw OBB (thin baseline as detected in the heatmap)
 *   Blue rectangle   — actual crop region fed to the recognizer (expandUp/expandDown)
 *   Red dot          — baseline centre (cx, cy)
 */

const path  = require('path');
const sharp = require('sharp');
const { KrakenSegmenter } = require('./src/segmenter');

const [,, imgArg, modelArg, outArg] = process.argv;

if (!imgArg) {
  console.error('Usage: node debug_segmentation.js <image> [seg_model] [output]');
  process.exit(1);
}

const imgPath   = path.resolve(imgArg);
const modelPath = path.resolve(modelArg  || 'tests/fixtures/segmentation.js_mlmodel');
const outPath   = path.resolve(outArg    || 'debug_segmentation.png');

function cropCorners(obb, lineHeight, topline) {
  const { cx, cy, angle, w: obbW } = obb;
  const expandUp   = topline ? lineHeight * 0.35 : lineHeight * 0.85;
  const expandDown = topline ? lineHeight * 0.85 : lineHeight * 0.35;
  const hPad = Math.ceil(lineHeight * 0.1);
  const hw   = obbW / 2 + hPad;
  const cosA = Math.cos(angle), sinA = Math.sin(angle);
  const vx = sinA, vy = -cosA;
  return [
    [cx - hw * cosA + expandUp   * vx, cy - hw * sinA + expandUp   * vy],
    [cx + hw * cosA + expandUp   * vx, cy + hw * sinA + expandUp   * vy],
    [cx + hw * cosA - expandDown * vx, cy + hw * sinA - expandDown * vy],
    [cx - hw * cosA - expandDown * vx, cy - hw * sinA - expandDown * vy],
  ];
}

function estimateLineHeight(lines, imageSize) {
  if (lines.length < 2) return Math.round(imageSize.height / 30);
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

(async () => {
  console.log(`Model  : ${modelPath}`);
  console.log(`Image  : ${imgPath}`);

  const segmenter = await KrakenSegmenter.create(modelPath);
  const { lines, imageSize } = await segmenter.segment(imgPath);
  const topline    = segmenter._meta.topline || false;
  const lineHeight = estimateLineHeight(lines, imageSize);

  console.log(`Found  : ${lines.length} lines  (${imageSize.width}×${imageSize.height})`);
  console.log(`LineH  : ${lineHeight}px  (expandUp=${Math.round(lineHeight * (topline ? 0.35 : 0.85))}px  expandDown=${Math.round(lineHeight * (topline ? 0.85 : 0.35))}px)`);

  const svgShapes = lines.map(({ obb, type }) => {
    // Red OBB polygon (raw baseline)
    const pts = obb.corners.map(([x, y]) => `${x},${y}`).join(' ');
    const obbPoly = `<polygon points="${pts}" stroke="#e03020" stroke-width="1.5" fill="rgba(220,48,32,0.06)"/>`;

    // Baseline centre dot
    const dot = `<circle cx="${Math.round(obb.cx)}" cy="${Math.round(obb.cy)}" r="3" fill="#e03020" opacity="0.9"/>`;

    // Blue oriented crop polygon (actual region sent to recognizer)
    const cc = cropCorners(obb, lineHeight, topline);
    const cpts = cc.map(([x, y]) => `${x},${y}`).join(' ');
    const cropPoly = `<polygon points="${cpts}" stroke="#2060e0" stroke-width="1.5" fill="rgba(32,96,224,0.06)"/>`;

    return obbPoly + dot + cropPoly;
  });

  const svg = Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${imageSize.width}" height="${imageSize.height}">
      ${svgShapes.join('\n      ')}
    </svg>`
  );

  await sharp(imgPath)
    .toColorspace('srgb')
    .composite([{ input: svg, blend: 'over' }])
    .png()
    .toFile(outPath);

  console.log(`Output : ${outPath}`);
  console.log(`\nLegend : red polygon = OBB (baseline)   blue rectangle = crop sent to OCR\n`);

  // Summary table
  console.log('  #   cx     cy     w      h     angle  crop-h  type');
  console.log('  ────────────────────────────────────────────────────────');
  lines.forEach(({ obb, type }, i) => {
    const deg = (obb.angle * 180 / Math.PI).toFixed(1);
    const cc  = cropCorners(obb, lineHeight, topline);
    const rys = cc.map(c => c[1]);
    const ch  = Math.ceil(Math.max(...rys)) - Math.floor(Math.min(...rys));
    console.log(
      `  ${String(i + 1).padStart(3)}  ` +
      `${String(Math.round(obb.cx)).padStart(5)}  ` +
      `${String(Math.round(obb.cy)).padStart(5)}  ` +
      `${String(Math.round(obb.w)).padStart(5)}  ` +
      `${String(Math.round(obb.h)).padStart(5)}  ` +
      `${deg.padStart(6)}°  ` +
      `${String(ch).padStart(6)}px  ${type}`
    );
  });
})().catch(e => { console.error(e); process.exit(1); });
