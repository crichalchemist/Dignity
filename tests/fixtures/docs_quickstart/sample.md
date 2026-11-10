# Sample

An unmarked fence is prose, not a command to run:

```bash
rm -rf everything
```

```bash quickstart
# train first
python -m train.cli --config config/smoke.yaml

python -m export.to_onnx --checkpoint runs/smoke/train/best.pt \
    --output dignity_risk.onnx
```
