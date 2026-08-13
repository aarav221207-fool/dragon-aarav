import importlib.util
import inspect
import json
import math
import os
import sys
import torch


def find_bdh_module():
  for root, dirs, files in os.walk("."):
    if "bdh.py" in files:
      return os.path.join(root, "bdh.py")
    if "bdh" in dirs and os.path.exists(
        os.path.join(root, "bdh", "__init__.py")
    ):
      return os.path.join(root, "bdh")
  return None


bdh_path = find_bdh_module()
if not bdh_path:
  sys.exit("ERROR: Could not find bdh module.")

if bdh_path.endswith(".py"):
  spec = importlib.util.spec_from_file_location("bdh", bdh_path)
  bdh = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(bdh)
else:
  sys.path.insert(0, os.path.dirname(bdh_path))
  import bdh

model_class = next(
    (
        getattr(bdh, c)
        for c in ["BDH", "BDHModel", "DragonHatchling", "Model", "Transformer"]
        if hasattr(bdh, c)
    ),
    None,
)
config_class = next(
    (
        getattr(bdh, c)
        for c in [
            "BDHConfig",
            "Config",
            "ModelConfig",
            "BDHConfigSmall",
            "BDHConfigBase",
        ]
        if hasattr(bdh, c)
    ),
    None,
)

if not model_class:
  sys.exit("ERROR: Could not find model class.")

OUTPUT_FILE = "bdh_microscope_rich.json"


def capture_tensor(t, name):
  """Capture summary metrics without full tensor payloads to keep JSON small."""
  if t is None or not isinstance(t, torch.Tensor):
    return None

  t = t.detach().cpu()
  t_float = t.float() if not (t.is_floating_point() or t.is_complex()) else t
  flat = t.flatten()
  flat_float = t_float.flatten()
  n = flat.numel()

  if n == 0:
    return {"name": name, "shape": list(t.shape), "numel": 0}

  mean_val = float(flat_float.mean()) if n > 0 else 0.0
  std_val = float(flat_float.std()) if n > 1 else 0.0

  return {
      "name": name,
      "shape": list(t.shape),
      "numel": n,
      "mean": 0.0 if math.isnan(mean_val) else mean_val,
      "std": 0.0 if math.isnan(std_val) else std_val,
      "min": float(flat_float.min()),
      "max": float(flat_float.max()),
      "sparsity": float((flat == 0).float().mean()),
      "sample": flat[: min(10, n)].tolist(),
  }


# Reduced to 3 key prompt samples to keep output super compact
TEST_INPUTS = [
    "The cat sat",
    "Machine learning is",
    "The quick brown fox jumps over the lazy dog",
]

config = config_class() if config_class else None
model = model_class(config) if config else model_class()
model.eval()

device = (
    next(model.parameters()).device
    if list(model.parameters())
    else torch.device("cpu")
)
sig = inspect.signature(model.forward)
param_names = list(sig.parameters.keys())

all_results = []

for idx, text in enumerate(TEST_INPUTS):
  tokens = [ord(c) for c in text]
  input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

  captures = []
  attention_weights = []
  hook_handles = []

  def make_hook(name, mod_type):
    def hook(m, inp, out):
      cap = {"module_path": name, "type": mod_type}

      if isinstance(out, torch.Tensor):
        cap["output"] = capture_tensor(out, f"{name}_out")
      elif isinstance(out, (tuple, list)):
        cap["output"] = (
            capture_tensor(out[0], f"{name}_out")
            if len(out) > 0 and isinstance(out[0], torch.Tensor)
            else None
        )

      if (
          isinstance(inp, tuple)
          and len(inp) > 0
          and isinstance(inp[0], torch.Tensor)
      ):
        cap["input"] = capture_tensor(inp[0], f"{name}_in")

      captures.append(cap)

    return hook

  for name, module in model.named_modules():
    if name:
      hook_handles.append(
          module.register_forward_hook(
              make_hook(name, module.__class__.__name__)
          )
      )

  with torch.no_grad():
    if "input_ids" in param_names:
      output = model(input_ids=input_tensor)
    elif "x" in param_names:
      output = model(x=input_tensor)
    elif "idx" in param_names:
      output = model(idx=input_tensor)
    else:
      output = model(input_tensor)

    final = output[0] if isinstance(output, (tuple, list)) else output

  for h in hook_handles:
    h.remove()

  all_results.append({
      "text": text,
      "tokens": tokens,
      "token_count": len(tokens),
      "captures": captures,
      "final_output": capture_tensor(final, "final") if final is not None else None,
  })

output_data = {
    "version": "6.0-light",
    "model": type(model).__name__,
    "total_params": sum(p.numel() for p in model.parameters()),
    "num_inputs": len(all_results),
    "inputs": all_results,
}

with open(OUTPUT_FILE, "w") as f:
  json.dump(output_data, f, indent=2, default=str)

size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"SUCCESS: Saved {OUTPUT_FILE} ({size_mb:.2f} MB)")
