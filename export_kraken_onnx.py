#!/usr/bin/env python3
"""
Export a Kraken .mlmodel or .safetensors model to .js_mlmodel.

A .js_mlmodel is a ZIP archive containing:
  model.onnx     — ONNX graph with dynamic batch and width axes
  metadata.json  — model-type-specific config (codec, class_mapping, etc.)

Supports both recognition and segmentation models.

Usage:
    env/bin/python3 export_kraken_onnx.py <model_path> [output_path]

If output_path is omitted, the .js_mlmodel is written next to the source file.
"""
import json
import zipfile
import tempfile
import argparse
from pathlib import Path

import torch
import torch.nn as nn


class _RecognitionExportWrapper(nn.Module):
    """Strips seq_lens and squeezes the H=1 output dim for CTC recognition models."""
    def __init__(self, inner_nn):
        super().__init__()
        self.inner_nn = inner_nn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.inner_nn(x, None)
        # out: (N, C, 1, W) → (N, C, W)
        return out.squeeze(2)


class _SegmentationExportWrapper(nn.Module):
    """Strips seq_lens and applies sigmoid for segmentation heatmap models."""
    def __init__(self, inner_nn):
        super().__init__()
        self.inner_nn = inner_nn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.inner_nn(x, None)
        # out: (N, C, H, W) — apply sigmoid so JS receives probabilities
        return torch.sigmoid(out)


def _export_recognition(model, model_path, output_path):
    _, channels, height, _ = model.input
    pad = model.user_metadata.get('hyper_params', {}).get('pad', 16)
    one_channel_mode = model.user_metadata.get('one_channel_mode', 'L')
    vgsl = model.user_metadata.get('vgsl', '')
    codec = model.codec.c2l

    print(f'  Type : recognition')
    print(f'  VGSL : {vgsl}')
    print(f'  Input: channels={channels}, height={height}, pad={pad}')
    print(f'  Codec: {len(codec)} entries')

    wrapper = _RecognitionExportWrapper(model.nn)
    wrapper.eval()
    dummy_input = torch.zeros(1, channels, height, 800)

    dynamic_axes = {
        'input':  {0: 'batch', 3: 'width'},
        'output': {0: 'batch', 2: 'width'},
    }
    metadata = {
        'model_type': 'recognition',
        'height': height,
        'channels': channels,
        'pad': pad,
        'one_channel_mode': one_channel_mode,
        'vgsl': vgsl,
        'codec': codec,
    }
    return wrapper, dummy_input, dynamic_axes, metadata


def _export_segmentation(model, model_path, output_path):
    _, channels, height, _ = model.input
    one_channel_mode = model.user_metadata.get('one_channel_mode', None)
    vgsl = model.user_metadata.get('vgsl', '')
    class_mapping = model.user_metadata.get('class_mapping', {})
    topline = model.user_metadata.get('topline', False)

    print(f'  Type : segmentation')
    print(f'  VGSL : {vgsl}')
    print(f'  Input: channels={channels}, height={height}')
    print(f'  Classes: {class_mapping}')

    wrapper = _SegmentationExportWrapper(model.nn)
    wrapper.eval()
    dummy_input = torch.zeros(1, channels, height, 800)

    dynamic_axes = {
        'input':  {0: 'batch', 3: 'width'},
        'output': {0: 'batch', 2: 'height', 3: 'width'},
    }
    metadata = {
        'model_type': 'segmentation',
        'height': height,
        'channels': channels,
        'one_channel_mode': one_channel_mode,
        'class_mapping': class_mapping,
        'topline': topline,
        'vgsl': vgsl,
    }
    return wrapper, dummy_input, dynamic_axes, metadata


def export(model_path: str, output_path: str | None = None) -> Path:
    from kraken.models.loaders import load_models

    model_path = Path(model_path)
    if output_path is None:
        output_path = model_path.with_suffix('.js_mlmodel')
    else:
        output_path = Path(output_path)

    print(f'Loading {model_path} …')
    # Try recognition first, then segmentation
    models = load_models(str(model_path), tasks=['recognition'])
    if not models:
        models = load_models(str(model_path), tasks=['segmentation'])
    if not models:
        raise ValueError(f'No supported model found in {model_path}')
    model = models[0]
    model.eval()

    model_types = model.user_metadata.get('model_type', [])
    if isinstance(model_types, str):
        model_types = [model_types]

    if 'segmentation' in model_types:
        wrapper, dummy_input, dynamic_axes, metadata = _export_segmentation(model, model_path, output_path)
    else:
        wrapper, dummy_input, dynamic_axes, metadata = _export_recognition(model, model_path, output_path)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / 'model.onnx'
        meta_path = Path(tmp) / 'metadata.json'

        print('Exporting ONNX …')
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(onnx_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
        )
        print(f'  Written {onnx_path.stat().st_size // 1024} KB')

        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

        print(f'Packing {output_path} …')
        with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(onnx_path, 'model.onnx')
            zf.write(meta_path, 'metadata.json')

    print(f'Done → {output_path}  ({output_path.stat().st_size // 1024} KB)')
    return output_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', help='Path to .mlmodel or .safetensors')
    p.add_argument('output', nargs='?', help='Output .js_mlmodel path (default: same name)')
    args = p.parse_args()
    export(args.model, args.output)


if __name__ == '__main__':
    main()
