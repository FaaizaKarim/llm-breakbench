"""LLM-BreakBench runner.

Evaluates every (model x prompt-variant) combination against the
adversarial task suite, reusing the llm-eval-harness library for
generation, sandboxed execution, and failure-mode classification.

Usage:
    python runner.py --models llama3.2 --base-url http://localhost:11434/v1
    python runner.py --models llama3.2 qwen2.5-coder:3b --prompts baseline cot-visible

Outputs one JSON file per combination into results/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

from llm_eval.client import OpenAICompatibleClient
from llm_eval.harness import load_tasks
from llm_eval.models import EvalResult, FailureMode
from llm_eval.report import summarize
from llm_eval.scorer import score

_CODE_BLOCKS_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str, mode: str = "first") -> str:
    """Pull Python source from a model reply.

    'first' takes the first fenced block (code-only prompts);
    'last' takes the final block (chain-of-thought prompts, where the
    model is instructed to end with its answer).
    """
    blocks = _CODE_BLOCKS_RE.findall(text)
    if blocks:
        block = blocks[0] if mode == "first" else blocks[-1]
        return block.strip()
    return text.strip()


class PromptedClient(OpenAICompatibleClient):
    """OpenAI-compatible client with a configurable system prompt."""

    def __init__(self, model: str, base_url: str, system_prompt: str,
                 timeout_s: float = 300.0) -> None:
        super().__init__(model=model, base_url=base_url, timeout_s=timeout_s)
        self.system_prompt = system_prompt

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self.name,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        headers = (
            {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        )
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


async def generate_with_retry(client: PromptedClient, prompt: str,
                              retries: int = 6) -> str:
    """Retry on HTTP 429 (rate limit) with the server's suggested backoff.

    Free API tiers throttle aggressively; without this, a benchmark run
    silently degrades into a wall of generation errors.
    """
    for attempt in range(retries):
        try:
            return await client.generate(prompt)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < retries - 1:
                wait = exc.response.headers.get("retry-after")
                delay = min(float(wait), 90.0) if wait else float(2 ** attempt)
                print(f"    rate limited; retrying in {delay:.0f}s", flush=True)
                await asyncio.sleep(delay + 1.0)
                continue
            raise
    raise RuntimeError("unreachable")


async def run_combo(client: PromptedClient, tasks, extract_mode: str,
                    sandbox_timeout: float, delay_s: float = 0.0) -> list[EvalResult]:
    """Evaluate all tasks sequentially (kind to CPU-bound local models)."""
    results: list[EvalResult] = []
    for task in tasks:
        if delay_s:
            await asyncio.sleep(delay_s)
        try:
            reply = await generate_with_retry(client, task.prompt)
            code = extract_code(reply, extract_mode)
        except Exception as exc:
            results.append(EvalResult(
                task_id=task.task_id, model=client.name, passed=False,
                failure_mode=FailureMode.SANDBOX_ERROR, tests_passed=0,
                tests_total=len(task.test_cases), duration_s=0.0,
                error_trace=f"generation error: {exc!r}",
            ))
            print(f"  {task.task_id}: generation error", flush=True)
            continue
        result = await score(code, task, client.name, timeout_s=sandbox_timeout)
        results.append(result)
        mark = "PASS" if result.passed else f"FAIL ({result.failure_mode.value})"
        print(f"  {task.task_id}: {mark} "
              f"[{result.tests_passed}/{result.tests_total}]", flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the BreakBench ablation grid.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--suite", default="suites/traps.json")
    parser.add_argument("--prompt-file", default="prompts/prompts.json")
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="Variant names to run (default: all)")
    parser.add_argument("--gen-timeout", type=float, default=300.0)
    parser.add_argument("--sandbox-timeout", type=float, default=10.0)
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between requests (rate-limit friendly)")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    tasks = load_tasks(args.suite)
    variants = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
    if args.prompts:
        variants = [v for v in variants if v["name"] in args.prompts]
    if not variants:
        print("No matching prompt variants.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    for model in args.models:
        for variant in variants:
            print(f"\n=== {model} x {variant['name']} "
                  f"({len(tasks)} tasks) ===", flush=True)
            client = PromptedClient(
                model=model, base_url=args.base_url,
                system_prompt=variant["system"], timeout_s=args.gen_timeout,
            )
            results = asyncio.run(run_combo(
                client, tasks, variant.get("extract", "first"),
                args.sandbox_timeout, args.delay,
            ))
            payload = {
                "model": model,
                "prompt_variant": variant["name"],
                "summary": summarize(results),
                "results": [r.to_dict() for r in results],
            }
            safe_model = model.replace(":", "_").replace("/", "_")
            out_path = out_dir / f"{safe_model}__{variant['name']}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            s = payload["summary"]
            print(f"--> {s['passed']}/{s['tasks']} passed "
                  f"({s['pass_rate']:.0%}) -> {out_path}", flush=True)

    print("\nAll combinations done. Next: python analyze.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
