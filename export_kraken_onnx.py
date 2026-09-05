#!/usr/bin/env python3
"""
Export a Kraken .mlmodel or .safetensors model to .js_mlmodel.

A .js_mlmodel is a ZIP archive containing:
  model.onnx     — ONNX graph with dynamic batch and width axes
  metadata.json  — model-type-specific config (codec, class_mapping, etc.)

Supports VGSL recognition, VGSL segmentation, and PP-OCRv6 recognition models.

Usage:
    env/bin/python3 export_kraken_onnx.py <model_path> [output_path]

If output_path is omitted, the .js_mlmodel is written next to the source file.
"""
import json
import types
import contextlib
import zipfile
import tempfile
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# Parity tolerances: (max abs diff, mean abs diff). The point of the check is to
# catch shapes baked into the graph by the tracer — those blow the difference up
# by orders of magnitude — not bit-exactness. Recurrent VGSL stacks accumulate
# visible float drift on the random inputs used here, so their max is loose while
# the mean stays tight; the feed-forward PP-OCR graph is held to a tight max.
_VGSL_TOL = {'max': 5e-2, 'mean': 1e-3}
_PPOCR_TOL = {'max': 1e-3, 'mean': 1e-4}


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


class _PPOCRExportWrapper(nn.Module):
    """
    Keeps seq_lens as a real graph input for PP-OCRv6 recognizers.

    The LightSVTR neck mixes across the whole sequence with global attention, so
    the padded columns of a batch must be masked out — which kraken does from
    ``seq_lens``. Dropping the argument (as the VGSL wrapper does) would corrupt
    every short line in a mixed-width batch.
    """
    def __init__(self, inner_nn):
        super().__init__()
        self.inner_nn = inner_nn

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor):
        out, out_lens = self.inner_nn(x, seq_lens)
        # out: (N, C, 1, W) → (N, C, W)
        return out.squeeze(2), out_lens


def _export_recognition(model, model_path, output_path):
    _, channels, height, _ = model.input
    pad = model.user_metadata.get('hyper_params', {}).get('pad', 16)
    one_channel_mode = model.user_metadata.get('one_channel_mode', 'L')
    vgsl = model.user_metadata.get('vgsl', '')
    codec = model.codec.c2l

    print(f'  Type : recognition (VGSL)')
    print(f'  VGSL : {vgsl}')
    print(f'  Input: channels={channels}, height={height}, pad={pad}')
    print(f'  Codec: {len(codec)} entries')

    wrapper = _RecognitionExportWrapper(model.nn)
    wrapper.eval()
    dummy_inputs = (torch.zeros(1, channels, height, 800),)

    dynamic_axes = {
        'input':  {0: 'batch', 3: 'width'},
        'output': {0: 'batch', 2: 'width'},
    }
    metadata = {
        'model_type': 'recognition',
        'architecture': 'vgsl',
        'height': height,
        'channels': channels,
        'pad': pad,
        'one_channel_mode': one_channel_mode,
        'vgsl': vgsl,
        'codec': codec,
    }
    spec = {
        'wrapper': wrapper,
        'dummy_inputs': dummy_inputs,
        'input_names': ['input'],
        'output_names': ['output'],
        'dynamic_axes': dynamic_axes,
        'metadata': metadata,
        'check_shapes': [(2, channels, height, 320), (2, channels, height, 800)],
        'tol': _VGSL_TOL,
    }
    return spec


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
    dummy_inputs = (torch.zeros(1, channels, height, 800),)

    dynamic_axes = {
        'input':  {0: 'batch', 3: 'width'},
        'output': {0: 'batch', 2: 'height', 3: 'width'},
    }
    metadata = {
        'model_type': 'segmentation',
        'architecture': 'vgsl',
        'height': height,
        'channels': channels,
        'one_channel_mode': one_channel_mode,
        'class_mapping': class_mapping,
        'topline': topline,
        'vgsl': vgsl,
    }
    spec = {
        'wrapper': wrapper,
        'dummy_inputs': dummy_inputs,
        'input_names': ['input'],
        'output_names': ['output'],
        'dynamic_axes': dynamic_axes,
        'metadata': metadata,
        # segmentation graphs are large; a single smaller width is enough
        'check_shapes': [(1, channels, height, 640)],
        'tol': _VGSL_TOL,
    }
    return spec


@contextlib.contextmanager
def _traceable_ppocr_lengths():
    """
    Swap kraken's ``_lengths_and_mask`` for a trace-safe equivalent.

    The original computes ``w_out / float(w_in)``; that ``float()`` freezes the
    traced input width into the graph, so ``out_lens`` — and with it the
    attention mask — comes out wrong at every other width. The replacement keeps
    the same arithmetic but leaves both widths symbolic.
    """
    from kraken.lib.ppocr import network as _net

    def _lengths_and_mask(seq_lens, w_in, w_out, device):
        lens = seq_lens.to(device=device, dtype=torch.float32)
        out_lens = (lens * w_out / w_in).floor().long().clamp(min=1)
        w_out_t = torch.as_tensor(w_out).to(device=device, dtype=out_lens.dtype)
        out_lens = torch.minimum(out_lens, w_out_t)
        positions = torch.arange(w_out, device=device)
        mask = positions[None, :] < out_lens[:, None]
        return out_lens, mask

    orig = _net._lengths_and_mask
    _net._lengths_and_mask = _lengths_and_mask
    try:
        yield
    finally:
        _net._lengths_and_mask = orig


def _freeze_backbone_pool(backbone, channels, height):
    """
    Make the PPLCNetV4 final pooling kernel a Python constant.

    ``PPLCNetV4.forward`` pools with ``kernel_size=(x.shape[2], 2)``; under the
    JIT tracer that height becomes a graph value and ONNX rejects the non-constant
    kernel. The model's input height is fixed (only the width is dynamic), so the
    feature height is resolved once here and baked in. Returns the original bound
    forward so a parity check can compare against the unpatched model.
    """
    with torch.no_grad():
        x = torch.zeros(1, channels, height, 64)
        x = backbone.conv1(x)
        for name in ('blocks2', 'blocks3', 'blocks4', 'blocks5', 'blocks6'):
            x = getattr(backbone, name)(x)
        feat_h = int(x.shape[2])

    orig_forward = backbone.forward

    def forward(self, x, _h=feat_h):
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)
        return F.avg_pool2d(x, kernel_size=(_h, 2))

    backbone.forward = types.MethodType(forward, backbone)
    return orig_forward


def _export_ppocr(model, model_path, output_path):
    from kraken.lib.ppocr.network import WIDTH_SUBSAMPLING

    _, channels, height, _ = model.input
    pad = model.user_metadata.get('hyper_params', {}).get('pad', 16)
    one_channel_mode = model.user_metadata.get('one_channel_mode', None)
    variant = model.user_metadata.get('variant')
    num_classes = model.user_metadata.get('num_classes')
    codec = model.codec.c2l

    print(f'  Type : recognition (PP-OCRv6)')
    print(f'  Variant: {variant}, classes={num_classes}')
    print(f'  Input: channels={channels}, height={height}, pad={pad}')
    print(f'  Codec: {len(codec)} entries')

    orig_backbone_forward = _freeze_backbone_pool(model.nn.backbone, channels, height)

    wrapper = _PPOCRExportWrapper(model.nn)
    wrapper.eval()
    # batch 2 with unequal lengths so both the batch axis and the masked branch
    # of the SVTR neck are traced
    dummy_inputs = (torch.zeros(2, channels, height, 800),
                    torch.tensor([800, 384], dtype=torch.long))

    dynamic_axes = {
        'input':    {0: 'batch', 3: 'width'},
        'seq_lens': {0: 'batch'},
        'output':   {0: 'batch', 2: 'width'},
        'out_lens': {0: 'batch'},
    }
    metadata = {
        'model_type': 'recognition',
        'architecture': 'ppocrv6',
        'variant': variant,
        'height': height,
        'channels': channels,
        'pad': pad,
        'one_channel_mode': one_channel_mode,
        'num_classes': num_classes,
        'width_subsampling': WIDTH_SUBSAMPLING,
        'seq_lens_input': True,
        'vgsl': '',
        'codec': codec,
    }
    spec = {
        'wrapper': wrapper,
        'dummy_inputs': dummy_inputs,
        'input_names': ['input', 'seq_lens'],
        'output_names': ['output', 'out_lens'],
        'dynamic_axes': dynamic_axes,
        'metadata': metadata,
        'check_shapes': [(2, channels, height, 320), (2, channels, height, 800)],
        # compare ONNX against the *unpatched* torch model, so the frozen pooling
        # kernel is validated rather than assumed
        'ref_fn': _unpatched_ref(model.nn, model.nn.backbone, orig_backbone_forward),
        'trace_patch': _traceable_ppocr_lengths,
        'tol': _PPOCR_TOL,
    }
    return spec


def _unpatched_ref(inner_nn, backbone, orig_forward):
    """Reference forward that temporarily restores the original backbone forward."""
    def ref(x, seq_lens):
        patched = backbone.forward
        backbone.forward = orig_forward
        try:
            out, _ = inner_nn(x, seq_lens)
        finally:
            backbone.forward = patched
        return out.squeeze(2)
    return ref


def _make_check_inputs(spec, shape):
    """Build a (torch args, ORT feeds) pair for one parity-check shape."""
    n, c, h, w = shape
    x = torch.rand(n, c, h, w)
    args = [x]
    feeds = {spec['input_names'][0]: x.numpy()}
    if 'seq_lens' in spec['input_names']:
        # unequal lengths: the second sample is deliberately short
        lens = torch.tensor([w] + [max(w // 2, 8)] * (n - 1), dtype=torch.long)
        args.append(lens)
        feeds['seq_lens'] = lens.numpy()
    return tuple(args), feeds


def _parity_check(spec, onnx_path):
    """
    Compare torch and onnxruntime outputs at several widths.

    Catches shapes baked into the graph by the JIT tracer (the SVTR attention
    reshapes from ``x.shape`` and the backbone pools with a shape-derived
    kernel), which would silently break every width but the one exported.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print('  onnxruntime not installed — skipping parity check')
        return True

    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    tol = spec.get('tol', _PPOCR_TOL)
    ok = True
    for shape in spec['check_shapes']:
        args, feeds = _make_check_inputs(spec, shape)
        ref_fn = spec.get('ref_fn', spec['wrapper'])
        with torch.no_grad():
            ref = ref_fn(*args)
        ref = ref[0] if isinstance(ref, tuple) else ref
        got = sess.run([spec['output_names'][0]], feeds)[0]
        if tuple(got.shape) != tuple(ref.shape):
            print(f'  ✗ {shape}: shape mismatch torch={tuple(ref.shape)} onnx={tuple(got.shape)}')
            ok = False
            continue
        delta = (torch.from_numpy(got) - ref).abs()
        dmax, dmean = float(delta.max()), float(delta.mean())
        passed = dmax <= tol['max'] and dmean <= tol['mean']
        status = '✓' if passed else '✗'
        print(f'  {status} {shape}: max |Δ| = {dmax:.3e}, mean |Δ| = {dmean:.3e} '
              f'(tol {tol["max"]:.0e}/{tol["mean"]:.0e})')
        if not passed:
            ok = False
    return ok


def export(model_path: str, output_path: str | None = None, check: bool = True) -> Path:
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

    if model.__class__.__name__ == 'PPOCRv6Model':
        spec = _export_ppocr(model, model_path, output_path)
    elif 'segmentation' in model_types:
        spec = _export_segmentation(model, model_path, output_path)
    else:
        spec = _export_recognition(model, model_path, output_path)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / 'model.onnx'
        meta_path = Path(tmp) / 'metadata.json'

        print('Exporting ONNX …')
        with spec.get('trace_patch', contextlib.nullcontext)():
            torch.onnx.export(
                spec['wrapper'],
                spec['dummy_inputs'],
                str(onnx_path),
                input_names=spec['input_names'],
                output_names=spec['output_names'],
                dynamic_axes=spec['dynamic_axes'],
                opset_version=17,
                dynamo=False,
            )
        print(f'  Written {onnx_path.stat().st_size // 1024} KB')

        if check:
            print('Checking torch/onnx parity …')
            if not _parity_check(spec, onnx_path):
                raise RuntimeError('ONNX parity check failed — the exported graph does '
                                   'not match the torch model at all tested widths.')

        meta_path.write_text(json.dumps(spec['metadata'], ensure_ascii=False, indent=2))

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
    p.add_argument('--no-check', dest='check', action='store_false',
                   help='Skip the torch/onnxruntime parity check after export')
    args = p.parse_args()
    export(args.model, args.output, check=args.check)


if __name__ == '__main__':
    main()
