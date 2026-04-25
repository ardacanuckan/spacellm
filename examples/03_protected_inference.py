"""03_protected_inference.py, protect a real HuggingFace LLM against a bit-flip.

Walks through, end-to-end:

1. Load GPT-2 (124 M params; downloads ~500 MB from the HuggingFace Hub
   on first run, then cached).
2. Generate the clean baseline output.
3. Make two more copies. Harden one with SpaceLLM's SelectiveTMR.
4. Inject the *same* exponent-bit flip into both copies, into the
   unprotected layer's weight tensor in one, into a single TMR replica
   in the other.
5. Generate from each. Show that:
   * the unprotected copy is destroyed by the flip,
   * the hardened copy reproduces the clean baseline because
     the TMR vote masks the corrupt replica.

Run with:

    uv run python examples/03_protected_inference.py
"""

from __future__ import annotations

import copy
import sys

import torch

import spacellm as sl
from spacellm._internal.bitops import flip_bit

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:  # pragma: no cover
    print("This example needs the [hf] extras: pip install 'spacellm[hf]'")
    sys.exit(1)


PROMPT = "The first satellite to carry a language model"
MAX_NEW_TOKENS = 20

# A flip we can predict: bit 30 of an FP32 element is the second-MSB of the
# exponent, toggling it shifts the value by ~2^128 and almost always
# produces a NaN or extreme magnitude in the resulting logits.
TARGET_BIT_POSITION = 30
TARGET_FLAT_INDEX = 0


def _generate(
    model: GPT2LMHeadModel,
    tok: GPT2Tokenizer,
    inputs: dict[str, torch.Tensor],
) -> str:
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def main() -> None:
    print("[1/5] loading gpt2 from the HuggingFace Hub...")
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    inputs = tok(PROMPT, return_tensors="pt")

    print("[2/5] generating the clean baseline...")
    baseline_text = _generate(base, tok, inputs)

    print("[3/5] cloning two copies and hardening one with SelectiveTMR(10%)...")
    unprotected = copy.deepcopy(base).eval()
    hardened = copy.deepcopy(base).eval()
    strategy = sl.protection.SelectiveTMR(top_k_percent=10.0)
    sl.harden(hardened, strategies=[strategy])

    if not strategy.wrapped_paths:
        print("[error] No nn.Linear modules eligible for TMR, cannot demo on this model.")
        sys.exit(1)

    target = strategy.wrapped_paths[0]
    print(f"        TMR-wrapped target layer: {target}")

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
    unp_text = _generate(unprotected, tok, inputs)
    hard_text = _generate(hardened, tok, inputs)

    print("\n" + "=" * 72)
    print(f"PROMPT:                                {PROMPT!r}")
    print("=" * 72)
    print(f"\nCLEAN BASELINE (no fault):\n  {baseline_text!r}")
    print(f"\nFAULT, NO PROTECTION:\n  {unp_text!r}")
    print(f"\nFAULT + SelectiveTMR (10% of nn.Linear):\n  {hard_text!r}")
    print()

    matches_baseline = hard_text == baseline_text
    unp_diverged = unp_text != baseline_text

    if matches_baseline and unp_diverged:
        print("[PASS] Protection masked the fault. The hardened model reproduced")
        print("       the clean baseline; the unprotected model diverged.")
    elif matches_baseline and not unp_diverged:
        print("[?]    Protection matched the baseline, but the unprotected copy")
        print("       wasn't visibly affected by this single flip. Try a more")
        print("       aggressive bit (closer to bit 31) or a different layer.")
    else:
        print("[!]    Hardened output differs from baseline, investigate.")
        print("       This usually means the chosen layer wasn't on the hot path")
        print("       for this prompt; try a different TARGET_FLAT_INDEX or layer.")


if __name__ == "__main__":
    main()
