"""Export Dignity models to ONNX format for deployment."""

import torch
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np


def export_to_onnx(model: torch.nn.Module,
                   output_path: str,
                   input_shape: Tuple[int, int, int] = (1, 100, 9),
                   opset_version: int = 13,
                   verify: bool = True) -> None:
    """
    Export Dignity model to ONNX format.
    
    Args:
        model: Trained Dignity model
        output_path: Path to save ONNX model
        input_shape: Input shape (batch, seq_len, features)
        opset_version: ONNX opset version
        verify: Whether to verify exported model
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Determine output names based on task
    output_names = ["predictions", "attention_weights"]
    
    print(f"Exporting model to {output_path}...")
    print(f"Input shape: {input_shape}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence_length"},
            "predictions": {0: "batch_size"},
            "attention_weights": {0: "batch_size", 1: "sequence_length"}
        }
    )
    
    print(f"Model exported successfully to {output_path}")
    
    # Verify export
    if verify:
        print("\nVerifying ONNX model...")
        verify_onnx_export(model, output_path, dummy_input)


def verify_onnx_export(pytorch_model: torch.nn.Module,
                       onnx_path: str,
                       test_input: torch.Tensor,
                       rtol: float = 1e-3,
                       atol: float = 1e-5) -> bool:
    """
    Verify that ONNX model produces same outputs as PyTorch model.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to exported ONNX model
        test_input: Test input tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if verification passes
    """
    # Check ONNX model validity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
        if isinstance(pytorch_output, tuple):
            pytorch_pred, pytorch_attn = pytorch_output
        else:
            pytorch_pred = pytorch_output
            pytorch_attn = None
    
    # Get ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare predictions
    pred_match = np.allclose(
        pytorch_pred.numpy(),
        ort_outputs[0],
        rtol=rtol,
        atol=atol
    )
    
    if pred_match:
        print("✓ Predictions match between PyTorch and ONNX")
        max_diff = np.max(np.abs(pytorch_pred.numpy() - ort_outputs[0]))
        print(f"  Max difference: {max_diff:.6e}")
    else:
        print("✗ Predictions do NOT match!")
        return False
    
    # Compare attention weights if available
    if pytorch_attn is not None and len(ort_outputs) > 1:
        attn_match = np.allclose(
            pytorch_attn.numpy(),
            ort_outputs[1],
            rtol=rtol,
            atol=atol
        )
        
        if attn_match:
            print("✓ Attention weights match between PyTorch and ONNX")
        else:
            print("✗ Attention weights do NOT match!")
            return False
    
    print("\n✓ Verification successful!")
    return True


def get_onnx_model_info(onnx_path: str) -> Dict:
    """
    Get information about exported ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        Dictionary with model information
    """
    onnx_model = onnx.load(onnx_path)
    
    info = {
        'graph_name': onnx_model.graph.name,
        'opset_version': onnx_model.opset_import[0].version,
        'inputs': [],
        'outputs': [],
        'nodes': len(onnx_model.graph.node)
    }
    
    # Input info
    for inp in onnx_model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                for dim in inp.type.tensor_type.shape.dim]
        info['inputs'].append({
            'name': inp.name,
            'shape': shape,
            'type': inp.type.tensor_type.elem_type
        })
    
    # Output info
    for out in onnx_model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic'
                for dim in out.type.tensor_type.shape.dim]
        info['outputs'].append({
            'name': out.name,
            'shape': shape,
            'type': out.type.tensor_type.elem_type
        })
    
    return info


def benchmark_onnx_inference(onnx_path: str,
                             input_shape: Tuple[int, int, int] = (1, 100, 9),
                             num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark ONNX model inference speed.
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Input shape for testing
        num_runs: Number of inference runs
        
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    # Create session
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    
    # Create random input
    test_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = ort_session.run(None, {input_name: test_input})
    
    # Benchmark
    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = ort_session.run(None, {input_name: test_input})
        end = time.perf_counter()
        timings.append((end - start) * 1000)  # Convert to ms
    
    timings = np.array(timings)
    
    return {
        'mean_ms': np.mean(timings),
        'std_ms': np.std(timings),
        'min_ms': np.min(timings),
        'max_ms': np.max(timings),
        'median_ms': np.median(timings),
        'p95_ms': np.percentile(timings, 95)
    }


if __name__ == '__main__':
    import argparse
    from models.dignity import Dignity
    from core.config import DignityConfig
    
    parser = argparse.ArgumentParser(description="Export Dignity model to ONNX")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output ONNX file path')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file (if not in checkpoint)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference benchmark')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = DignityConfig.from_yaml(args.config)
    else:
        # Try to infer from checkpoint name
        config = DignityConfig()  # Use defaults
    
    # Load model
    print("Loading model...")
    model = Dignity(
        task=config.model.task,
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        n_layers=config.model.n_layers,
        dropout=config.model.dropout
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Export
    export_to_onnx(
        model=model,
        output_path=args.output,
        input_shape=(1, config.data.seq_len, len(config.data.features))
    )
    
    # Get info
    print("\nModel Information:")
    info = get_onnx_model_info(args.output)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Benchmark
    if args.benchmark:
        print("\nRunning inference benchmark...")
        stats = benchmark_onnx_inference(args.output)
        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  Std: {stats['std_ms']:.2f} ms")
        print(f"  Median: {stats['median_ms']:.2f} ms")
        print(f"  P95: {stats['p95_ms']:.2f} ms")
