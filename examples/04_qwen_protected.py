"""04_qwen_protected.py, protect a real Qwen LLM against a bit-flip.

Same idea as ``examples/03_protected_inference.py`` but on Qwen rather
than GPT-2. Qwen is the model SBFA (arXiv:2509.21843) used to drive
MMLU from 71% to 0% with a single bit-flip, the threat
SpaceLLM is built to defend against.

Walks through, end-to-end:

1. Load Qwen/Qwen2.5-0.5B-Instruct (~1 GB; downloads on first run,
   then cached). 0.5 B params is small enough to run on a developer
   laptop in CPU mode under a minute.
2. Generate the clean baseline output for a fixed chat prompt.
3. Make two more copies. Harden one with SpaceLLM's SelectiveTMR.
4. Inject the *same* exponent-bit flip into both copies, into the
   unprotected layer's weight tensor in one, into a single TMR
   replica in the other.
5. Generate from each. Show that:
   * the unprotected copy is destroyed by the flip,
   * the hardened copy reproduces the clean baseline because the
     TMR vote masks the corrupt replica.

Run with:

    uv run python examples/04_qwen_protected.py
"""

from __future__ import annotations

import copy
import sys

import torch

import spacellm as sl
from spacellm._internal.bitops import flip_bit

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    print("This example needs the [hf] extras: pip install 'spacellm[hf]'")
    sys.exit(1)


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
USER_PROMPT = "In one sentence, why is radiation dangerous for AI hardware in space?"
MAX_NEW_TOKENS = 60

# A flip we can predict: bit 30 of an FP32 element is the second-MSB of
# the exponent, toggling it shifts the value by ~2^128 and almost
# always produces a NaN or extreme magnitude in the resulting logits.
TARGET_BIT_POSITION = 30
TARGET_FLAT_INDEX = 0


def _generate(
    model,
    tok: AutoTokenizer,
    prompt_text: str,
) -> str:
    inputs = tok(prompt_text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    response_ids = out[0][inputs.input_ids.shape[1] :]
    return tok.decode(response_ids, skip_special_tokens=True).strip()


def main() -> None:
    print(f"[1/5] loading {MODEL_ID} from the HuggingFace Hub...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()

    messages = [{"role": "user", "content": USER_PROMPT}]
    prompt_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    print("[2/5] generating the clean baseline...")
    baseline_text = _generate(base, tok, prompt_text)

    print("[3/5] cloning two copies and hardening one with SelectiveTMR(5%)...")
    unprotected = copy.deepcopy(base).eval()
    hardened = copy.deepcopy(base).eval()
    strategy = sl.protection.SelectiveTMR(top_k_percent=5.0)
    sl.harden(hardened, strategies=[strategy])

    if not strategy.wrapped_paths:
        print("[error] No nn.Linear modules eligible for TMR, cannot demo on this model.")
        sys.exit(1)

    target = strategy.wrapped_paths[0]  # the largest Linear in the model
    print(f"        TMR-wrapped target layer: {target}")
    print(f"        ({len(strategy.wrapped_paths)} layers wrapped in total)")

    print(
        f"[4/5] flipping bit {TARGET_BIT_POSITION} of element {TARGET_FLAT_INDEX} "
        f"of '{target}.weight' in BOTH copies\n"
        f"        (unprotected: hits the live weight; hardened: hits replica A only)...",
    )

    unp_layer = unprotected.get_submodule(target)
    flip_bit(unp_layer.weight.data, TARGET_FLAT_INDEX, TARGET_BIT_POSITION)

    hard_layer = hardened.get_submodule(target)
    # On the hardened copy, only one of the three replicas is corrupted.
    # The TMR vote (median of three) masks it.
    flip_bit(hard_layer.weight_a.data, TARGET_FLAT_INDEX, TARGET_BIT_POSITION)

    print("[5/5] generating from each copy...")
    unp_text = _generate(unprotected, tok, prompt_text)
    hard_text = _generate(hardened, tok, prompt_text)

    bar = "=" * 78
    print("\n" + bar)
    print(f"PROMPT:  {USER_PROMPT!r}")
    print(bar)
    print(f"\nCLEAN BASELINE (no fault):\n  {baseline_text!r}")
    print(f"\nFAULT, NO PROTECTION:\n  {unp_text!r}")
    print(f"\nFAULT + SelectiveTMR (5% of nn.Linear):\n  {hard_text!r}")
    print()

    matches_baseline = hard_text == baseline_text
    unp_diverged = unp_text != baseline_text

    if matches_baseline and unp_diverged:
        print("[PASS] Protection masked the fault. The hardened model reproduced")
        print("       the clean baseline; the unprotected model diverged.")
    elif matches_baseline and not unp_diverged:
        print("[?]    Protection matched the baseline, but the unprotected copy")
        print("       wasn't visibly affected by this flip.")
    else:
        print("[!]    Hardened output differs from baseline, investigate.")


if __name__ == "__main__":
    main()
