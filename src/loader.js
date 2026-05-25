'use strict';
const AdmZip = require('adm-zip');

/**
 * Load a .js_mlmodel file (ZIP containing model.onnx + metadata.json).
 * Returns { onnxBytes: Buffer, metadata: object }.
 */
function loadJsMlmodel(filePath) {
  const zip = new AdmZip(filePath);
  const onnxEntry = zip.getEntry('model.onnx');
  const metaEntry = zip.getEntry('metadata.json');
  if (!onnxEntry) throw new Error(`${filePath}: missing model.onnx`);
  if (!metaEntry) throw new Error(`${filePath}: missing metadata.json`);
  return {
    onnxBytes: zip.readFile(onnxEntry),
    metadata: JSON.parse(zip.readAsText(metaEntry)),
  };
}

module.exports = { loadJsMlmodel };
