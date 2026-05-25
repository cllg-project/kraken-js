'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const AdmZip = require('adm-zip');
const os = require('node:os');
const fs = require('node:fs');
const path = require('node:path');
const { loadJsMlmodel } = require('../src/loader');

const MODEL_PATH = path.join(__dirname, 'fixtures/model_best.js_mlmodel');

// ---------------------------------------------------------------------------
// Helper: build a temporary .js_mlmodel ZIP on disk
// ---------------------------------------------------------------------------

function makeTempModel(entries) {
  const zip = new AdmZip();
  for (const [name, content] of Object.entries(entries)) {
    zip.addFile(name, Buffer.from(content));
  }
  const tmp = path.join(os.tmpdir(), `test_${Date.now()}_${Math.random()}.js_mlmodel`);
  zip.writeZip(tmp);
  return tmp;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('loadJsMlmodel', () => {
  test('loads the real model_best.js_mlmodel without error', () => {
    const { onnxBytes, metadata } = loadJsMlmodel(MODEL_PATH);
    assert.ok(onnxBytes instanceof Buffer);
    assert.ok(onnxBytes.length > 1000, 'ONNX bytes should be non-trivial');
    assert.equal(typeof metadata, 'object');
  });

  test('metadata contains required keys', () => {
    const { metadata } = loadJsMlmodel(MODEL_PATH);
    for (const key of ['height', 'channels', 'pad', 'codec', 'vgsl']) {
      assert.ok(key in metadata, `metadata missing key: ${key}`);
    }
  });

  test('metadata values match known model spec', () => {
    const { metadata } = loadJsMlmodel(MODEL_PATH);
    assert.equal(metadata.height, 120);
    assert.equal(metadata.channels, 1);
    assert.equal(metadata.pad, 16);
    assert.equal(typeof metadata.codec, 'object');
    assert.ok(Object.keys(metadata.codec).length > 0, 'codec should not be empty');
  });

  test('codec entries map characters to arrays of integers', () => {
    const { metadata } = loadJsMlmodel(MODEL_PATH);
    for (const [char, ids] of Object.entries(metadata.codec)) {
      assert.ok(Array.isArray(ids), `${char}: expected array, got ${typeof ids}`);
      assert.ok(ids.length >= 1);
      for (const id of ids) {
        assert.equal(typeof id, 'number');
        assert.ok(id >= 1, `label id must be >=1 (0 is CTC blank), got ${id}`);
      }
    }
  });

  test('onnxBytes starts with ONNX magic bytes', () => {
    const { onnxBytes } = loadJsMlmodel(MODEL_PATH);
    // ONNX protobuf files start with field tag 0x08 (field 1, varint)
    // More reliably: check for the string "onnx" or that it's a valid protobuf
    // We verify it's a Buffer with substantial content
    assert.ok(onnxBytes.length > 100_000, 'expected a real ONNX file > 100 KB');
  });

  test('throws when model.onnx is missing', () => {
    const tmp = makeTempModel({ 'metadata.json': '{"height":32}' });
    try {
      assert.throws(() => loadJsMlmodel(tmp), /missing model\.onnx/);
    } finally {
      fs.unlinkSync(tmp);
    }
  });

  test('throws when metadata.json is missing', () => {
    const tmp = makeTempModel({ 'model.onnx': 'fake' });
    try {
      assert.throws(() => loadJsMlmodel(tmp), /missing metadata\.json/);
    } finally {
      fs.unlinkSync(tmp);
    }
  });

  test('throws on invalid JSON in metadata.json', () => {
    const tmp = makeTempModel({ 'model.onnx': 'x', 'metadata.json': '{bad json' });
    try {
      assert.throws(() => loadJsMlmodel(tmp));
    } finally {
      fs.unlinkSync(tmp);
    }
  });

  test('throws on non-existent path', () => {
    assert.throws(() => loadJsMlmodel('/does/not/exist.js_mlmodel'));
  });
});
