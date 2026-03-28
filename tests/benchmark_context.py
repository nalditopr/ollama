"""
Benchmark Qwen3.5 35B-A3B at various context lengths.
Measures prompt processing speed, generation speed, and verifies NIAH.
Tests sparse V skip effectiveness (skip rate increases with context).
"""

import json
import time
import urllib.request
import sys

OLLAMA_HOST = "http://127.0.0.1:11435"
MODEL = "qwen3.5:35b-a3b"

NEEDLE = "The secret password is DELTA-7749-XRAY."
PADDING = (
    "Standard operating procedures section covering enterprise deployment guidelines. "
    "All production systems must be fully documented and reviewed before deployment. "
    "Quality assurance testing follows the standardized protocol. "
    "Scheduled maintenance windows occur on the first Sunday of each month. "
    "Infrastructure provisioning is managed through Terraform. "
)


def make_haystack(num_docs):
    """Build a haystack with a needle buried in the middle."""
    chunks = [f"Document {i+1}. {PADDING}" for i in range(num_docs)]
    chunks[num_docs // 2] = f"Document {num_docs//2 + 1}. CLASSIFIED: {NEEDLE}"
    prompt = " ".join(chunks) + " What is the secret password? State only the password value."
    return prompt


def run_benchmark(prompt, num_predict=200, num_ctx=262144):
    """Send prompt to model and return timing stats."""
    data = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "think": False,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "temperature": 0.0,
        }
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
    wall_time = time.time() - t0

    pt = result.get("prompt_eval_count", 0)
    pd = result.get("prompt_eval_duration", 1)
    gt = result.get("eval_count", 0)
    gd = result.get("eval_duration", 1)
    ld = result.get("load_duration", 0)

    full_text = str(result.get("response", "")) + str(result.get("thinking", ""))
    niah = "DELTA-7749-XRAY" in full_text

    return {
        "prompt_tokens": pt,
        "prompt_tok_s": pt / (pd / 1e9) if pd > 0 else 0,
        "prompt_time": pd / 1e9,
        "gen_tokens": gt,
        "gen_tok_s": gt / (gd / 1e9) if gd > 0 else 0,
        "gen_time": gd / 1e9,
        "load_time": ld / 1e9,
        "wall_time": wall_time,
        "niah": niah,
    }


def warmup():
    """Warm up the model with a short prompt."""
    print("Warming up model...")
    data = json.dumps({
        "model": MODEL,
        "prompt": "Hi",
        "stream": False,
        "options": {"num_predict": 5, "num_ctx": 262144}
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        json.loads(resp.read())
    print("Warm up complete.\n")


def main():
    # Context sizes to test (number of padding documents)
    test_configs = [
        (50,    "~2K"),
        (200,   "~8K"),
        (500,   "~20K"),
        (1000,  "~40K"),
        (2000,  "~80K"),
        (3000,  "~120K"),
        (4000,  "~160K"),
    ]

    warmup()

    print("=" * 90)
    print(f"{'Context':>10} | {'Prompt Tokens':>14} | {'Prompt tok/s':>13} | {'Gen tok/s':>10} | {'NIAH':>5} | {'Wall Time':>10}")
    print("-" * 90)

    results = []
    for num_docs, label in test_configs:
        prompt = make_haystack(num_docs)
        est_tokens = len(prompt) // 4
        print(f"{label:>10} | {'...running':>14}", end="", flush=True)

        try:
            r = run_benchmark(prompt, num_predict=300, num_ctx=262144)
            print(f"\r{label:>10} | {r['prompt_tokens']:>14,} | {r['prompt_tok_s']:>10,.0f} t/s | {r['gen_tok_s']:>7,.0f} t/s | {'PASS' if r['niah'] else 'FAIL':>5} | {r['wall_time']:>8.1f}s")
            results.append((label, r))
        except Exception as e:
            print(f"\r{label:>10} | ERROR: {e}")
            break

    print("=" * 90)
    print()

    # Summary
    if results:
        print("Summary:")
        print(f"  Model:   {MODEL}")
        print(f"  KV type: q4_0 (set via OLLAMA_KV_CACHE_TYPE)")
        print(f"  Flash:   enabled")
        print(f"  Sparse V: enabled (threshold 1e-6)")
        print()
        print(f"  Min prompt speed: {min(r['prompt_tok_s'] for _, r in results):,.0f} tok/s")
        print(f"  Max prompt speed: {max(r['prompt_tok_s'] for _, r in results):,.0f} tok/s")
        print(f"  Min gen speed:    {min(r['gen_tok_s'] for _, r in results):,.0f} tok/s")
        print(f"  Max gen speed:    {max(r['gen_tok_s'] for _, r in results):,.0f} tok/s")
        niah_pass = sum(1 for _, r in results if r['niah'])
        print(f"  NIAH:             {niah_pass}/{len(results)} passed")


if __name__ == "__main__":
    main()
