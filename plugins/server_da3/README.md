# Depth Anything 3 webpolicy server

This server exposes the Depth Anything 3 model behind the webpolicy interface. It can load the model either from the packaged repository weights or directly from the Hugging Face Hub.

## Usage

```bash
python -m plugins.server_da3.da3 --host 0.0.0.0 --port 8080 \
  --model-source huggingface --hf-model-id depth-anything/da3nested-giant-large
```

Key options:

- `--model-source` — set to `huggingface` to pull weights from the Hub (default) or `repo` to use the raw repository weights that ship with the package. Any other string is forwarded directly to `DepthAnything3.from_pretrained` as a path or model id.
- `--hf-model-id` — Hugging Face model id to load when `--model-source` is `huggingface`.
- `--device` — override the compute device (defaults to CUDA when available, otherwise CPU).

The server expects `DA3Payload` request bodies as defined in [`da3.py`](./da3.py) and returns serialized `Prediction` objects.
