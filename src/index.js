'use strict';
const { KrakenRecognizer } = require('./recognizer');
const { MultiRecognizer } = require('./multirecognizer');
const { KrakenSegmenter } = require('./segmenter');
const { KrakenPipeline } = require('./pipeline');
const { DFineSegmenter } = require('./dfine');
const { ReadingOrder } = require('./readingorder');
const { merge, candidates } = require('./merge');
module.exports = { KrakenRecognizer, MultiRecognizer, KrakenSegmenter, KrakenPipeline, DFineSegmenter, ReadingOrder, merge, candidates };
