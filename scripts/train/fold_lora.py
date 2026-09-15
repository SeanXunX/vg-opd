from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM

BASE = os.environ.get("CROPD_FOLD_BASE", os.path.join(os.environ.get("CROPD_MODELS", "models"), "Qwen3-4B"))
PROBE = "model.layers.10.self_attn.q_proj.weight"


def probe(model_dir: str):
    for p in sorted(Path(model_dir).glob("model-*.safetensors")):
        with safe_open(str(p), framework="pt") as f:
            if PROBE in f.keys():
                return f.get_tensor(PROBE)
    raise KeyError(PROBE)


def fold(target: Path) -> None:
    adapter = target / "lora_adapter"
    assert adapter.is_dir(), f"{target} has no lora_adapter/; nothing to fold"
    base_t = probe(BASE)
    if not torch.allclose(base_t, probe(str(target))):
        print(f"{target.name}: weights already differ from the base; skipping (looks folded)")
        return
    print(f"{target.name}: folding ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter), torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
    model.save_pretrained(str(target), safe_serialization=True)
    del model
    merged_t = probe(str(target))
    delta = (base_t - merged_t).abs().max().item()
    assert delta > 0, f"{target.name}: weights still equal the base after folding"
    gc = json.loads((target / "generation_config.json").read_text())
    eos = gc.get("eos_token_id")
    assert 151645 in (eos if isinstance(eos, list) else [eos]), f"{target.name}: eos_token_id must include 151645 (im_end), got {eos}"
    print(f"{target.name}: OK, probe max|delta|={delta:.4g}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        fold(Path(d))
