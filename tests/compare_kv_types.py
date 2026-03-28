#!/usr/bin/env python3
"""Compare KV cache quality across different configurations.

Usage:
    1. Start ollama with a specific KV cache type:
       OLLAMA_KV_CACHE_TYPE=turbo3 ./ollama serve

    2. Run this script with a label:
       python tests/compare_kv_types.py --label turbo3

    3. Repeat steps 1-2 for other KV types (f16, q8_0, q4_0, etc.)

    4. Compare saved results:
       python tests/compare_kv_types.py --compare

Results are saved to tests/kv_results/<label>.json and can be compared later.
"""

import json
import time
import os
import sys
import urllib.request
import urllib.error
import argparse
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "kv_results"

# Needle-in-a-haystack template
NIAH_NEEDLE = "The secret passphrase is: QUANTUM-DOLPHIN-42."
NIAH_FILLER = (
    "The history of artificial intelligence dates back to ancient myths and stories "
    "of artificial beings endowed with intelligence. The field was founded as an "
    "academic discipline in 1956. Since then, AI has experienced several waves of "
    "optimism and disappointment, known as AI winters. In the 21st century, AI "
    "techniques became widespread due to advances in computing power and large "
    "datasets. Machine learning, particularly deep learning, has driven many recent "
    "advances. Neural networks with many layers can learn complex patterns in data. "
    "Transformers, introduced in 2017, revolutionized natural language processing. "
)


def build_niah_prompt(n_filler_words: int) -> tuple[str, str]:
    """Build a needle-in-a-haystack prompt with the needle buried at ~50% depth."""
    words = NIAH_FILLER.split()
    # Repeat filler to reach target word count
    filler_words = []
    while len(filler_words) < n_filler_words:
        filler_words.extend(words)
    filler_words = filler_words[:n_filler_words]

    midpoint = len(filler_words) // 2
    filler_before = " ".join(filler_words[:midpoint])
    filler_after = " ".join(filler_words[midpoint:])

    prompt = (
        f"/no_think Read the following document carefully and answer the question at the end.\n\n"
        f"--- DOCUMENT START ---\n"
        f"{filler_before}\n\n"
        f"{NIAH_NEEDLE}\n\n"
        f"{filler_after}\n"
        f"--- DOCUMENT END ---\n\n"
        f"What is the secret passphrase mentioned in the document? Reply with ONLY the passphrase."
    )
    expected = "QUANTUM-DOLPHIN-42"
    return prompt, expected


# Test definitions
TESTS = [
    {
        "name": "math_multiply",
        "prompt": "/no_think What is 17*23? Reply with just the number.",
        "expected": "391",
        "category": "factual",
    },
    {
        "name": "math_addition",
        "prompt": "/no_think What is 847 + 256? Reply with just the number.",
        "expected": "1103",
        "category": "factual",
    },
    {
        "name": "capital_france",
        "prompt": "/no_think What is the capital of France? Reply with just the city name.",
        "expected": "Paris",
        "category": "factual",
    },
    {
        "name": "capital_japan",
        "prompt": "/no_think What is the capital of Japan? Reply with just the city name.",
        "expected": "Tokyo",
        "category": "factual",
    },
    {
        "name": "code_reverse",
        "prompt": "/no_think Write a Python one-liner to reverse a string variable s. Reply with just the code.",
        "expected": "[::-1]",
        "category": "code",
    },
    {
        "name": "code_fizzbuzz",
        "prompt": '/no_think Write a Python one-liner list comprehension for FizzBuzz from 1 to 15. Reply with just the code.',
        "expected": "Fizz",
        "category": "code",
    },
    {
        "name": "reasoning",
        "prompt": "/no_think If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer Yes or No and explain in one sentence.",
        "expected": "No",
        "category": "reasoning",
    },
    # NIAH tests are built dynamically
    {"name": "niah_100", "prompt": None, "expected": None, "docs": 100, "category": "niah"},
    {"name": "niah_500", "prompt": None, "expected": None, "docs": 500, "category": "niah"},
    {"name": "niah_1000", "prompt": None, "expected": None, "docs": 1000, "category": "niah"},
]


def query_ollama(model: str, prompt: str, host: str = "http://localhost:11434") -> tuple[str, float]:
    """Send a prompt to ollama and return (response_text, latency_seconds)."""
    url = f"{host}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
        },
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return f"ERROR: {e}", time.perf_counter() - t0
    except TimeoutError:
        return "ERROR: timeout", time.perf_counter() - t0

    elapsed = time.perf_counter() - t0
    return body.get("response", ""), elapsed


def check_match(response: str, expected: str) -> bool:
    """Check if the expected substring appears in the response."""
    return expected.lower() in response.lower()


def run_tests(model: str, host: str, label: str) -> dict:
    """Run all tests and return results dict."""
    results = {
        "label": label,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": [],
        "summary": {},
    }

    passed = 0
    total = 0

    for test in TESTS:
        name = test["name"]
        prompt = test["prompt"]
        expected = test["expected"]

        # Build NIAH prompts dynamically
        if test.get("docs"):
            prompt, expected = build_niah_prompt(test["docs"])

        print(f"  Running {name}...", end=" ", flush=True)
        response, latency = query_ollama(model, prompt, host)

        if response.startswith("ERROR:"):
            match = False
            print(f"FAIL ({response})")
        else:
            match = check_match(response, expected)
            status = "PASS" if match else "FAIL"
            print(f"{status} ({latency:.2f}s)")

        total += 1
        if match:
            passed += 1

        results["tests"].append({
            "name": name,
            "category": test.get("category", ""),
            "passed": match,
            "latency_s": round(latency, 3),
            "expected": expected,
            "response_preview": response[:200].strip(),
        })

    results["summary"] = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
    }

    return results


def save_results(results: dict, label: str):
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{label}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


def compare_results():
    """Load and compare all saved results."""
    if not RESULTS_DIR.exists():
        print("No results directory found. Run some tests first.")
        return

    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("No result files found. Run some tests first.")
        return

    all_results = []
    for f in files:
        with open(f) as fp:
            all_results.append(json.load(fp))

    # Collect all test names
    test_names = []
    for r in all_results:
        for t in r["tests"]:
            if t["name"] not in test_names:
                test_names.append(t["name"])

    labels = [r["label"] for r in all_results]

    # Print header
    col_w = 14
    header = f"{'Test':<20}" + "".join(f"{l:>{col_w}}" for l in labels)
    print("\n" + "=" * len(header))
    print("KV Cache Type Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Print each test row
    for tname in test_names:
        row = f"{tname:<20}"
        for r in all_results:
            test_data = next((t for t in r["tests"] if t["name"] == tname), None)
            if test_data:
                status = "PASS" if test_data["passed"] else "FAIL"
                latency = test_data["latency_s"]
                cell = f"{status} {latency:.1f}s"
            else:
                cell = "N/A"
            row += f"{cell:>{col_w}}"
        print(row)

    # Print summary row
    print("-" * len(header))
    row = f"{'TOTAL PASS':<20}"
    for r in all_results:
        s = r["summary"]
        cell = f"{s['passed']}/{s['total']} ({s['pass_rate']}%)"
        row += f"{cell:>{col_w}}"
    print(row)

    # Print average latency
    row = f"{'Avg latency':<20}"
    for r in all_results:
        lats = [t["latency_s"] for t in r["tests"] if not t["response_preview"].startswith("ERROR")]
        avg = sum(lats) / len(lats) if lats else 0
        cell = f"{avg:.2f}s"
        row += f"{cell:>{col_w}}"
    print(row)

    print("=" * len(header))

    # Print per-category breakdown
    categories = []
    for r in all_results:
        for t in r["tests"]:
            if t["category"] and t["category"] not in categories:
                categories.append(t["category"])

    if categories:
        print(f"\n{'Category':<20}" + "".join(f"{l:>{col_w}}" for l in labels))
        print("-" * len(header))
        for cat in categories:
            row = f"{cat:<20}"
            for r in all_results:
                cat_tests = [t for t in r["tests"] if t["category"] == cat]
                cat_passed = sum(1 for t in cat_tests if t["passed"])
                cat_total = len(cat_tests)
                cell = f"{cat_passed}/{cat_total}" if cat_total > 0 else "N/A"
                row += f"{cell:>{col_w}}"
            print(row)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare KV cache quality across different Ollama configurations."
    )
    parser.add_argument(
        "--model", default="qwen3.5:35b-a3b",
        help="Model name to test (default: qwen3.5:35b-a3b)"
    )
    parser.add_argument(
        "--label", default=None,
        help="Label for this run (e.g., turbo3, f16, q8_0). Required unless --compare is used."
    )
    parser.add_argument(
        "--host", default="http://localhost:11434",
        help="Ollama API host (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare previously saved results instead of running tests."
    )

    args = parser.parse_args()

    if args.compare:
        compare_results()
        return

    if not args.label:
        print("Error: --label is required when running tests.")
        print("Example: python tests/compare_kv_types.py --label turbo3")
        sys.exit(1)

    print(f"KV Cache Quality Test")
    print(f"  Model: {args.model}")
    print(f"  Label: {args.label}")
    print(f"  Host:  {args.host}")
    print()

    results = run_tests(args.model, args.host, args.label)
    save_results(results, args.label)

    # Print quick summary
    s = results["summary"]
    print(f"\nSummary: {s['passed']}/{s['total']} passed ({s['pass_rate']}%)")


if __name__ == "__main__":
    main()
