#!/usr/bin/env python
"""
ONNX Export, Verification & Laptop Evaluation for Feathered Insights.
Exports trained Swin Transformer to ONNX FP32 (opset 18, dynamic axes)
and benchmarks CPU latency under the 400ms target.
"""

import argparse
import json
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from myapp.ml.classifier import CUBModel


def export_to_onnx(
    pth_path: Path,
    classes_path: Path,
    output_path: Path,
    image_size: int = 224,
    opset_version: int = 18,
) -> None:
    """Exports PyTorch Swin Transformer model to ONNX FP32 format."""
    print("=" * 60)
    print("[EXPORT] Exporting Swin Transformer to ONNX FP32")
    print("=" * 60)

    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"[OK] Loaded {num_classes} species class names.")

    if not pth_path.exists():
        print(f"[ERROR] PyTorch weights not found at {pth_path}")
        sys.exit(1)

    device = torch.device("cpu")
    model = CUBModel(num_classes=num_classes, pretrained=False).to(device)
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[OK] PyTorch weights loaded from {pth_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size, requires_grad=False)

    print(f"[INFO] Exporting to: {output_path} (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )
    print(f"[OK] ONNX export complete! Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    verify_onnx(output_path, dummy_input, model)


def verify_onnx(onnx_path: Path, dummy_input: torch.Tensor, pytorch_model) -> None:
    """Verifies ONNX model numerical equivalence with PyTorch."""
    try:
        import onnxruntime as ort
        print("\n[INFO] Validating numerical equivalence with ONNX Runtime...")
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            torch_out = pytorch_model(dummy_input).numpy()
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_out = session.run(None, ort_inputs)[0]
        diff = np.max(np.abs(torch_out - ort_out))
        print(f"[OK] Maximum absolute difference: {diff:.6f}")
        if diff < 1e-3:
            print("[SUCCESS] Models are numerically equivalent (max_diff < 1e-3)!")
        else:
            print("[WARNING] Difference exceeds tolerance!")
    except Exception as e:
        print(f"[WARNING] Verification skipped: {e}")


def preprocess_image(image_path: Path, image_size: int = 224) -> np.ndarray:
    """Loads and preprocesses image for ONNX inference."""
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0).astype(np.float32)


def run_inference(onnx_path: Path, classes_path: Path, image_path: Path, image_size: int = 224, top_k: int = 5):
    """Runs ONNX inference on a single image and displays top-k predictions."""
    import onnxruntime as ort
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_tensor = preprocess_image(image_path, image_size)
    start = time.perf_counter()
    ort_out = session.run(None, {session.get_inputs()[0].name: input_tensor})[0]
    elapsed = (time.perf_counter() - start) * 1000
    probs = torch.nn.functional.softmax(torch.from_numpy(ort_out[0]), dim=0).numpy()
    top_indices = np.argsort(probs)[::-1][:top_k]

    print(f"\n[RESULT] Inference time: {elapsed:.1f} ms")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {class_names[idx]}: {probs[idx]*100:.2f}%")
    return class_names[top_indices[0]], float(probs[top_indices[0]])


def benchmark_latency(onnx_path: Path, image_size: int = 224, iterations: int = 50, warmup: int = 5) -> None:
    """Benchmarks ONNX Runtime CPU latency across multiple runs."""
    import onnxruntime as ort
    print(f"\n[BENCHMARK] ONNX CPU Latency ({iterations} iterations, {warmup} warmup)")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    print(f"\n[BENCHMARK RESULTS]\n  Mean:   {np.mean(arr):.2f} ms\n  Median: {np.median(arr):.2f} ms")
    print(f"  Min:    {np.min(arr):.2f} ms\n  Max:    {np.max(arr):.2f} ms")
    print(f"  P95:    {np.percentile(arr, 95):.2f} ms\n  P99:    {np.percentile(arr, 99):.2f} ms")
    target_ms = 400.0
    status = "PASS" if np.mean(arr) < target_ms else "FAIL"
    print(f"\n[STATUS] Sub-{target_ms:.0f}ms target: [{status}]")


def main():
    parser = argparse.ArgumentParser(description="ONNX Export & Evaluation for Feathered Insights")
    subparsers = parser.add_subparsers(dest="command", required=True)

    exp = subparsers.add_parser("export", help="Export PyTorch model to ONNX")
    exp.add_argument("--model", required=True)
    exp.add_argument("--classes", required=True)
    exp.add_argument("--output", required=True)
    exp.add_argument("--image-size", type=int, default=224)
    exp.add_argument("--opset", type=int, default=18)

    inf = subparsers.add_parser("infer", help="Run inference on an image")
    inf.add_argument("--model", required=True)
    inf.add_argument("--classes", required=True)
    inf.add_argument("--image", required=True)
    inf.add_argument("--image-size", type=int, default=224)
    inf.add_argument("--top-k", type=int, default=5)

    bnc = subparsers.add_parser("benchmark", help="Benchmark ONNX CPU latency")
    bnc.add_argument("--model", required=True)
    bnc.add_argument("--image-size", type=int, default=224)
    bnc.add_argument("--iterations", type=int, default=50)
    bnc.add_argument("--warmup", type=int, default=5)

    args = parser.parse_args()
    if args.command == "export":
        export_to_onnx(Path(args.model), Path(args.classes), Path(args.output), args.image_size, args.opset)
    elif args.command == "infer":
        run_inference(Path(args.model), Path(args.classes), Path(args.image), args.image_size, args.top_k)
    elif args.command == "benchmark":
        benchmark_latency(Path(args.model), args.image_size, args.iterations, args.warmup)


if __name__ == "__main__":
    main()