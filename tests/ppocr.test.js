'use strict';
const { test, describe, before } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const sharp = require('sharp');
const { KrakenRecognizer } = require('../src/recognizer');
const { loadJsMlmodel } = require('../src/loader');

const MODEL   = path.join(__dirname, 'fixtures/ppocr.js_mlmodel');
const EXAMPLE = path.join(__dirname, 'fixtures/example_line.png');
const OOD     = path.join(__dirname, 'fixtures/ood_example.png');

// The fixture is ~57 MB and is not committed. Build it with:
//   npm run fixtures:ppocr
// The expected strings below are for the PP-OCRv6 model that script downloads
// (Zenodo record 22232579, ppocr_v6_tau090).
const HAS_MODEL = fs.existsSync(MODEL);

const EXPECTED_EXAMPLE_SUBSTR = 'ἀλλὰ ἀλόγῳ πάθει καὶ μάστιγι δαιμόνων'.normalize('NFD');
const EXPECTED_OOD_SUBSTR     = 'κολάζετε'.normalize('NFD');

describe('PP-OCRv6 recognition', { skip: HAS_MODEL ? false : 'run `npm run fixtures:ppocr` to enable' }, () => {
  let recognizer;

  before(async () => {
    recognizer = await KrakenRecognizer.create(MODEL);
  });

  // -------------------------------------------------------------------------
  // Metadata / graph signature
  // -------------------------------------------------------------------------

  describe('metadata', () => {
    test('describes a PP-OCRv6 recognition model', () => {
      const { metadata } = loadJsMlmodel(MODEL);
      assert.equal(metadata.model_type, 'recognition');
      assert.equal(metadata.architecture, 'ppocrv6');
      assert.equal(metadata.variant, 'medium');
      assert.equal(metadata.height, 96);
      assert.equal(metadata.channels, 3);
      assert.equal(metadata.pad, 16);
      assert.equal(metadata.width_subsampling, 8);
      assert.equal(metadata.seq_lens_input, true);
      assert.equal(metadata.num_classes, 1630);
    });

    test('graph takes seq_lens and returns out_lens', () => {
      assert.equal(recognizer._needsSeqLens, true);
      assert.deepEqual(recognizer._session.inputNames, ['input', 'seq_lens']);
      assert.deepEqual(recognizer._session.outputNames, ['output', 'out_lens']);
    });
  });

  // -------------------------------------------------------------------------
  // recognize
  // -------------------------------------------------------------------------

  describe('recognize', () => {
    let result;
    before(async () => { result = await recognizer.recognize(EXAMPLE); });

    test('transcribes the example line', () => {
      assert.ok(result.text.normalize('NFD').includes(EXPECTED_EXAMPLE_SUBSTR),
                `got: ${result.text}`);
    });

    test('transcribes an out-of-distribution line', async () => {
      const { text } = await recognizer.recognize(OOD);
      assert.ok(text.normalize('NFD').includes(EXPECTED_OOD_SUBSTR), `got: ${text}`);
    });

    test('chars are ordered left to right with sane confidences', () => {
      assert.ok(result.chars.length > 0);
      let prev = -1;
      for (const c of result.chars) {
        assert.ok(c.x0 >= prev, `x0 ${c.x0} went backwards from ${prev}`);
        assert.ok(c.x1 >= c.x0);
        assert.ok(c.conf > 0 && c.conf <= 1, `conf out of range: ${c.conf}`);
        prev = c.x0;
      }
    });

    test('text is the concatenation of chars', () => {
      assert.equal(result.text, result.chars.map(c => c.char).join(''));
    });
  });

  // -------------------------------------------------------------------------
  // Batching
  //
  // The model masks the batch padding out of its attention neck (that is what
  // the seq_lens input is for), but batching is still not bit-identical to
  // single-line inference: the backbone's SAME padding phase depends on the
  // parity of the batch width, so a line can shift by one input column. Kraken's
  // own torch model behaves the same way. Equality therefore only holds for a
  // uniform-width batch; for mixed widths we assert the transcription is
  // essentially unchanged, which is what caught the CTC stride bug where short
  // lines in a batch decoded from misaligned memory and came out as garbage.
  // -------------------------------------------------------------------------

  describe('recognizeBatch', () => {
    test('uniform-width batch matches single-image inference exactly', async () => {
      const single = await recognizer.recognize(EXAMPLE);
      const batch  = await recognizer.recognizeBatch([EXAMPLE, EXAMPLE, EXAMPLE]);
      assert.equal(batch.length, 3);
      for (const r of batch) {
        assert.equal(r.text, single.text);
        assert.deepEqual(r.chars, single.chars);
      }
    });

    test('a short line batched with a long one still decodes correctly', async () => {
      const { width, height } = await sharp(OOD).metadata();
      const short = await sharp(OOD)
        .extract({ left: 0, top: 0, width: Math.round(width / 4), height })
        .toBuffer();

      const [longSingle, shortSingle] = await Promise.all([
        recognizer.recognize(EXAMPLE),
        recognizer.recognize(short),
      ]);
      const batch = await recognizer.recognizeBatch([EXAMPLE, short]);

      assert.ok(similarity(batch[0].text, longSingle.text) > 0.9,
                `long line: ${batch[0].text} vs ${longSingle.text}`);
      assert.ok(similarity(batch[1].text, shortSingle.text) > 0.9,
                `short line: ${batch[1].text} vs ${shortSingle.text}`);
    });

    test('out_lens bounds each image to its own output width', async () => {
      const { width, height } = await sharp(OOD).metadata();
      const short = await sharp(OOD)
        .extract({ left: 0, top: 0, width: Math.round(width / 4), height })
        .toBuffer();
      const [long, brief] = await recognizer.recognizeBatch([EXAMPLE, short]);
      // The short crop must not pick up characters from the padded region.
      assert.ok(brief.chars.length < long.chars.length / 2,
                `short line decoded ${brief.chars.length} chars`);
    });
  });
});

/** Levenshtein similarity in [0,1]; 1 means identical. */
function similarity(a, b) {
  if (a === b) return 1;
  if (!a.length || !b.length) return 0;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return 1 - prev[b.length] / Math.max(a.length, b.length);
}
