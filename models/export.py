import torch

from config.config import ExportConfig
from models.agent_hybrid import DignityModel
from models.signal_policy import SignalPolicyAgent


def export_to_onnx(
        model: DignityModel,
        export_cfg: ExportConfig,
        example_input: torch.Tensor,
        task_type: str = "classification",
):
    model.eval()
    input_names = ["input"]
    output_names = ["logits" if task_type == "classification" else "predictions"]
    dynamic_axes = {"input": {0: "batch"}, output_names[0]: {0: "batch"}}

    torch.onnx.export(
        model,
        example_input,
        export_cfg.onnx_path,
        export_params=True,
        opset_version=export_cfg.opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported ONNX model to {export_cfg.onnx_path}")


def export_signal_policy_to_onnx(
        agent: SignalPolicyAgent,
        export_cfg: ExportConfig,
        example_input: torch.Tensor,
        task_type: str = "classification",
):
    agent.eval()
    input_names = ["input"]
    output_names = ["policy_logits", "value"]
    if task_type == "classification":
        output_names.append("direction_logits")
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    def _forward(x):
        out = agent(x, detach_signal=True)
        outputs = [out["policy_logits"], out["value"]]
        if task_type == "classification" and "direction_logits" in out["signal"]["aux"]:
            outputs.append(out["signal"]["aux"]["direction_logits"])
        return tuple(outputs)

    torch.onnx.export(
        _forward,
        example_input,
        export_cfg.onnx_path,
        export_params=True,
        opset_version=export_cfg.opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported signal+policy ONNX model to {export_cfg.onnx_path}")
