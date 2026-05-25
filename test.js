'use strict';
const path = require('path');
const { KrakenRecognizer } = require('./src/index');

const MODEL   = path.join(__dirname, 'tests/fixtures/model_best.js_mlmodel');
const LINE    = path.join(__dirname, 'tests/fixtures/example_line.png');

async function main() {
  console.log('Loading model …');
  const r = await KrakenRecognizer.create(MODEL);

  console.log('\n--- Single image ---');
  const t0 = Date.now();
  const { text, chars } = await r.recognize(LINE);
  console.log(`Result  : ${text}`);
  console.log(`Expected: ζετε, ἀλλὰ ἀλόγῳ πάθει καὶ μάστιγι δαιμόνων φαύλων ἐξε-`);
  console.log(`Time    : ${Date.now() - t0} ms`);
  console.log(`Chars   : ${chars.length}  (first 5: ${JSON.stringify(chars.slice(0, 5))})`);

  console.log('\n--- Batch (3×) ---');
  const t1 = Date.now();
  const batch = await r.recognizeBatch([LINE, LINE, LINE]);
  console.log(`Results : ${batch.map(b => b.text).join(' | ')}`);
  console.log(`Time    : ${Date.now() - t1} ms`);
}

main().catch(err => { console.error(err); process.exit(1); });
