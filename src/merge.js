'use strict';
/**
 * Multi-model CTC-spike merge (port of cllg-multimodel-run `spikes.py` + `merge.py`).
 *
 * Three recognisers read the same line: `base` (its text is kept), `new` (fine-tuned on rare editorial
 * signs) and `spec` (every letter masked to `_`). A sign enters the base text only where `new` AND
 * `spec` emit it at the same x. A confident base character is never overwritten (a logged conflict).
 * Signs only `new` reads are `candidates`; unexplained low-confidence base characters are `flags`.
 *
 * Pure functions, no dependencies. A spike is `{ ch, x, conf }`; `x` is in line-image pixels.
 */

const DASH = new Set([...'-‐‑–—']);
const DEFAULT_SIGNS = new Set([...'[]{}<>|+†«»']);
const DEFAULT_CONFUSABLE = new Set([...'[]()\'’`~"“”‘!lI1/\\.,;:·-–—_*^+ ']);

const DEFAULT_PARAMS = Object.freeze({
  tau: 0.9,        // base characters below this confidence may be replaced
  tol: 2.0,        // new/spec agreement tolerance, in character pitches
  win: 1.0,        // base window a sign may replace, in pitches (1.6x for ||)
  clear: 0.45,     // a confident base char closer than this (pitches) blocks an insertion
  candConf: 0.9,   // confidence for a new-only sign to be listed as candidate
  lowconf: 0.6,    // base spikes below this are flagged when unexplained
  specOnly: true,  // insert a confident `spec`-only sign into free space (extension to the Python rule)
  signs: DEFAULT_SIGNS,
  confusable: DEFAULT_CONFUSABLE,
});

/** Python-style round(): ties to even, so output matches the reference implementation. */
function pyRound(v, nd = 0) {
  const m = 10 ** nd;
  const s = v * m;
  const f = Math.floor(s);
  if (s - f === 0.5 && Number.isInteger(v * 1024)) return (f % 2 === 0 ? f : f + 1) / m; // exact tie (v dyadic, so v*m is exact): to even
  return Number(v.toFixed(nd));                              // otherwise correctly rounded, like Python
}

const isMark = ch => [...ch].length === 1 && /^\p{M}$/u.test(ch);
const isAlpha = ch => ch.length > 0 && /^\p{L}+$/u.test(ch);
const isSpace = ch => ch.length > 0 && /^\s+$/u.test(ch);

/** Recognizer `chars` (`{char, conf, x0, x1}`) -> spikes; x is the middle of the character's span. */
function spikes(chars) {
  return chars.map(({ char, conf, x0, x1 }) => ({ ch: char, x: (x0 + x1) / 2, conf }));
}

/** Mean advance between base (non-mark) characters, at least 4 px; 8 px for an empty line. */
function pitch(sp) {
  const xs = sp.filter(s => !isMark(s.ch)).map(s => s.x);
  if (xs.length < 2) return 8.0;
  return Math.max((xs[xs.length - 1] - xs[0]) / (xs.length - 1), 4.0);
}

/** Special-sign spikes; two `|` side by side (within 1.6 pitches) become one `||` token. */
function signTokens(sp, pt, signs) {
  const set = signs instanceof Set ? signs : new Set(signs);
  const toks = [];
  for (let i = 0; i < sp.length; i++) {
    const { ch, x, conf } = sp[i];
    if (!set.has(ch)) continue;
    if (ch === '|' && i + 1 < sp.length && sp[i + 1].ch === '|' && Math.abs(sp[i + 1].x - x) <= 1.6 * pt) {
      toks.push({ ch: '||', x: (x + sp[i + 1].x) / 2, conf: Math.min(conf, sp[i + 1].conf) });
      i++;
      continue;
    }
    toks.push({ ch, x, conf });
  }
  return toks;
}

function punctTokens(sp, signs) {
  return sp.filter(({ ch }) =>
    !isAlpha(ch) && !isSpace(ch) && !isMark(ch) && !signs.has(ch) && !DASH.has(ch) && ch !== '|');
}

function nearIdx(chars, x, win) {
  const out = [];
  chars.forEach((t, i) => {
    if (Math.abs(t.x - x) <= win && t.ch !== '' && t.ch !== ' ' && !isMark(t.ch)) out.push(i);
  });
  return out;
}

const countChar = (s, c) => [...s].filter(k => k === c).length;

/**
 * @param {Array<{ch,x,conf}>} base
 * @param {Array<{ch,x,conf}>} nw
 * @param {Array<{ch,x,conf}>} spec
 * @param {object} [params]  overrides of DEFAULT_PARAMS
 * @returns {{ text: string, edits: object[], flags: Array<[string, number, number]>,
 *   merged: Array<{ch, x, conf, src?}> }}  `merged` are the spikes of `text`, one per character
 */
function merge(base, nw, spec, params = {}) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const pt = pitch(base);
  const tol = P.tol * pt;
  const nt = signTokens(nw, pt, P.signs);
  const st = signTokens(spec, pt, P.signs);
  const agreed = nt.filter(t => st.some(u => u.ch === t.ch && Math.abs(u.x - t.x) <= tol));
  const chars = base.map((s, i) => ({ ...s, src: i })); // `src`: index in `base`, absent for inserted signs
  const edits = [];

  for (const { ch: c, x, conf: cf } of agreed) {
    const win = P.win * pt * (c === '||' ? 1.6 : 1.0);
    const near = nearIdx(chars, x, win);
    const have = near.map(i => chars[i].ch).join('');
    if ((c === '||' && (have.includes('||') || countChar(have, '|') >= 2)) || (c !== '||' && have.includes(c))) {
      continue; // the base already has it
    }
    const sub = near.filter(i => chars[i].conf < P.tau || P.confusable.has(chars[i].ch));
    const free = near.length === 0 || Math.min(...near.map(i => Math.abs(chars[i].x - x))) >= P.clear * pt;
    if (sub.length || free) {
      if (sub.length) {
        const haveSub = sub.map(i => chars[i].ch).join('');
        for (const i of sub.slice(1)) chars[i].ch = '';
        edits.push({ x: pyRound(x), from: haveSub, to: c, conf: [pyRound(chars[sub[0]].conf, 2), pyRound(cf, 2)] });
        chars[sub[0]].ch = c;
        chars[sub[0]].conf = cf;
      } else {
        let k = chars.findIndex(t => t.x > x);
        if (k < 0) k = chars.length;
        chars.splice(k, 0, { ch: c, x, conf: cf });
        edits.push({ x: pyRound(x), from: '', to: c, conf: [null, pyRound(cf, 2)] });
      }
    } else {
      edits.push({ x: pyRound(x), from: have, to: c, conflict: true });
    }
  }

  // spec-only signs: `spec` has no letter confusion, so where it is sure (>= candConf) of a sign that
  // `new` missed (blank at that frame), fill free base space. Never substitutes, never overwrites.
  if (P.specOnly) {
    for (const { ch: c, x, conf: cf } of st) {
      if (cf < P.candConf || agreed.some(a => a.ch === c && Math.abs(a.x - x) <= tol)) continue;
      const near = nearIdx(chars, x, P.win * pt * (c === '||' ? 1.6 : 1.0));
      const have = near.map(i => chars[i].ch).join('');
      if ((c === '||' && (have.includes('||') || countChar(have, '|') >= 2)) || (c !== '||' && have.includes(c))) continue;
      const free = near.length === 0 || Math.min(...near.map(i => Math.abs(chars[i].x - x))) >= P.clear * pt;
      if (free) {
        let k = chars.findIndex(t => t.x > x);
        if (k < 0) k = chars.length;
        chars.splice(k, 0, { ch: c, x, conf: cf });
        edits.push({ x: pyRound(x), from: '', to: c, conf: [null, pyRound(cf, 2)], kind: 'spec' });
      } else {
        edits.push({ x: pyRound(x), from: have, to: c, conflict: true, kind: 'spec' });
      }
    }
  }

  // second pass: punctuation / digits both models agree on, where exactly one low-confidence base spike sits
  const st2 = punctTokens(spec, P.signs);
  for (const { ch: c, x, conf: cf } of punctTokens(nw, P.signs)) {
    if (!st2.some(u => u.ch === c && Math.abs(u.x - x) <= tol)) continue;
    const near = nearIdx(chars, x, P.win * pt);
    if (near.some(i => chars[i].ch === c)) continue;
    const low = near.filter(i => chars[i].conf < P.tau
      && !([...chars[i].ch].length === 1 && isAlpha(chars[i].ch) && chars[i].conf >= 0.6));
    if (low.length === 1 && near.length === 1 && !DASH.has(chars[low[0]].ch)) {
      const t = chars[low[0]];
      edits.push({ x: pyRound(x), from: t.ch, to: c, conf: [pyRound(t.conf, 2), pyRound(cf, 2)], kind: 'punct' });
      t.ch = c;
      t.conf = cf;
    }
  }

  const text = chars.map(t => t.ch).join('');
  const flags = base
    .filter(b => b.conf < P.lowconf && b.ch !== ' ' && !isMark(b.ch)
      && !edits.some(e => Math.abs(b.x - e.x) <= P.win * pt))
    .map(b => [b.ch, pyRound(b.x), pyRound(b.conf, 2)]);
  return { text, edits, flags, merged: chars.filter(t => t.ch !== '') };
}

/** Signs only `new` reads (conf >= candConf), not applied, with none of that sign in the base nearby. */
function candidates(base, nw, applied, params = {}) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const pt = pitch(base);
  const done = applied.filter(e => P.signs.has(e.to) || e.to === '||').map(e => e.x);
  const out = [];
  for (const { ch: c, x, conf: cf } of signTokens(nw, pt, P.signs)) {
    if (cf < P.candConf || done.some(d => Math.abs(x - d) <= P.tol * pt)) continue;
    const near = base.filter(b => Math.abs(b.x - x) <= P.win * pt).map(b => b.ch).join('');
    if (near.includes(c) || (c === '||' && countChar(near, '|') >= 2)) continue;
    out.push({ x: pyRound(x), sign: c, conf: pyRound(cf, 2), base_near: near });
  }
  return out;
}

module.exports = {
  DEFAULT_PARAMS, DEFAULT_SIGNS, DEFAULT_CONFUSABLE,
  spikes, pitch, signTokens, isMark, merge, candidates, pyRound,
};
