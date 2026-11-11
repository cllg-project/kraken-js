'use strict';
const { KrakenRecognizer } = require('./recognizer');
const { KrakenSegmenter } = require('./segmenter');
const { KrakenPipeline } = require('./pipeline');
const { DFineSegmenter } = require('./dfine');
const { ReadingOrder } = require('./readingorder');
module.exports = { KrakenRecognizer, KrakenSegmenter, KrakenPipeline, DFineSegmenter, ReadingOrder };
