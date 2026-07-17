# LLM-BreakBench

**An adversarial Python benchmark + prompt-engineering ablation for code-generation models.**

24 hand-designed tasks, each hiding a classic Python trap that language models are known to fumble — mutable default arguments, closure late binding, float accumulation, iterator exhaustion, `str.strip` misuse, `lst[-0:]`, `zip` truncation, `True == 1`, exponential recursion, async result ordering, path traversal, and `eval` injection. Every model failure is classified, traced, and analyzed.

Built on top of my [llm-eval-harness](https://github.com/FaaizaKarim/llm-eval-harness) (async evaluation pipeline, sandboxed execution, failure-mode taxonomy) — BreakBench adds the adversarial task design and the prompt-ablation methodology.

## The question it answers

Everyone reports pass rates. BreakBench asks two sharper questions:

1. **Where exactly do code models break?** Not "67%," but *which trap families* — and with reproducible error traces for each failure.
2. **Does prompt engineering actually fix it?** The same suite runs under 4 system-prompt variants: `baseline` (code only), `edge-checklist` (explicit pitfall list), `cot-visible` (reason first, code last), and `adversarial-warning` ("this task contains a trap"). The delta between variants measures how much of a model's failure is *capability* vs *prompting*.

## Trap categories

state-aliasing · closures · float-precision · iterators · strings · sorting · exceptions · slicing · type-coercion · iteration/zip · regex · performance · async · security

Each task's test cases include a probe for the specific trap — e.g. the grid task doesn't just check dimensions, it mutates one cell and verifies row independence; the calculator task feeds `__import__("os")` to catch `eval`-based solutions behaviorally.

## Run it

```bash
# 1. install the harness (sibling repo) + this project's deps
pip install -e ../llm-eval-harness
pip install -r requirements.txt

# 2. run the ablation grid against a local model (Ollama)
python runner.py --models llama3.2 --base-url http://localhost:11434/v1

# multiple models, selected variants
python runner.py --models llama3.2 qwen2.5-coder:3b --prompts baseline edge-checklist

# 3. charts + report
python analyze.py
```

`runner.py` writes one JSON per (model × prompt) combination into `results/`, including per-task error traces. `analyze.py` produces `charts/pass_rate_by_prompt.png`, `charts/pass_rate_by_category.png`, and a `REPORT.md` with the ablation grid, a failure inventory, and an analysis section written by hand — because the point of an evaluation is the reasoning, not just the numbers.

## Findings

See [REPORT.md](REPORT.md) for the current results and written analysis.

## Design notes

- **Sequential execution** (`--concurrency` deliberately absent): local CPU inference degrades badly under parallel load, which contaminates timing-based failure modes.
- **Two extraction modes**: chain-of-thought prompts legitimately emit multiple code blocks; the runner takes the *last* block for CoT variants and the *first* otherwise — extraction policy is part of evaluation methodology, not a detail.
- **Behavioral security tests**: the suite never inspects source code for banned constructs; it proves misbehavior with inputs (an `eval`-based calculator *passes arithmetic* but *fails the injection probe*).
- **JSON-safe expectations**: expected values are JSON types only, and prompts explicitly request lists (not tuples) so equality checks measure logic, not serialization luck.

## Tech

Python 3.10+ · llm-eval-harness · asyncio · httpx · matplotlib · Ollama (or any OpenAI-compatible endpoint)
