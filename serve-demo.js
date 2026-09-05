#!/usr/bin/env node
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;
const ROOT = path.join(__dirname, 'docs');
const FIXTURES = path.join(__dirname, 'tests', 'fixtures');

const MIME = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.css':  'text/css',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
};

http.createServer((req, res) => {
  let urlPath = req.url.split('?')[0];
  if (urlPath === '/') urlPath = '/index.html';

  // /models/* → tests/fixtures/*
  const modelsPrefix = '/models/';
  let filePath;
  if (urlPath.startsWith(modelsPrefix)) {
    filePath = path.join(FIXTURES, urlPath.slice(modelsPrefix.length));
    if (!filePath.startsWith(FIXTURES)) { res.writeHead(403); res.end(); return; }
  } else {
    filePath = path.join(ROOT, urlPath);
    if (!filePath.startsWith(ROOT)) { res.writeHead(403); res.end(); return; }
  }

  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); res.end('Not found'); return; }
    const ext = path.extname(filePath);
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      // Explicit length so the demo can report model download progress; without
      // it Node falls back to chunked encoding and the loader has no total.
      'Content-Length': data.length,
      // Cross-origin isolation, which is what unlocks multi-threaded WASM in
      // ONNX Runtime. GitHub Pages cannot send these, so the deployed demo runs
      // single-threaded (still off the main thread — see ort.env.wasm.proxy).
      // Cross-origin scripts must therefore be loaded with crossorigin="anonymous";
      // see the CDN script tags in docs/index.html.
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    res.end(data);
  });
}).listen(PORT, () => {
  console.log(`Demo running at http://localhost:${PORT}`);
});
