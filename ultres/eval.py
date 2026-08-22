"""v1.6: Benchmark evaluation harness for UltRes.

Runs UltRes on standard benchmark datasets (HumanEval, MMLU) and scores
the output automatically. This gives an objective, reproducible way to
measure improvements across versions.

Usage:
    python -m ultres.eval --benchmark humaneval --n 20
    python -m ultres.eval --benchmark mmlu --n 50
    python -m ultres.eval --benchmark humaneval --n 164 --output results.json

Scoring:
    - HumanEval: pass@1 (does the generated function pass the test cases?)
    - MMLU: accuracy (does the model pick the correct A/B/C/D answer?)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Result of a single benchmark task."""
    task_id: str
    prompt: str
    expected: str
    generated: str = ""
    passed: bool = False
    error: str = ""
    elapsed: float = 0.0
    pages_researched: int = 0


@dataclass
class BenchmarkReport:
    """Aggregate results for a benchmark run."""
    benchmark: str
    total_tasks: int = 0
    passed: int = 0
    failed: int = 0
    results: list[TaskResult] = field(default_factory=list)
    total_elapsed: float = 0.0
    avg_pages_researched: float = 0.0

    @property
    def score(self) -> float:
        """Pass rate or accuracy as a percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.passed / self.total_tasks) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "total_tasks": self.total_tasks,
            "passed": self.passed,
            "failed": self.failed,
            "score_pct": round(self.score, 2),
            "total_elapsed": round(self.total_elapsed, 2),
            "avg_pages_researched": round(self.avg_pages_researched, 1),
            "results": [
                {
                    "task_id": r.task_id,
                    "passed": r.passed,
                    "error": r.error[:200] if r.error else "",
                    "elapsed": round(r.elapsed, 2),
                    "pages_researched": r.pages_researched,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# HumanEval benchmark
# ---------------------------------------------------------------------------

HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"


def download_humaneval(cache_dir: Path, n: int) -> list[dict[str, Any]]:
    """Download HumanEval dataset (or use cached copy).

    Returns list of task dicts with: task_id, prompt, test, entry_point, canonical_solution.
    """
    cache_file = cache_dir / "humaneval.jsonl"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not cache_file.exists():
        import gzip
        import httpx
        # Download the gzipped file.
        resp = httpx.get(HUMANEVAL_URL, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        # Decompress and save.
        decompressed = gzip.decompress(resp.content)
        cache_file.write_bytes(decompressed)

    tasks = []
    for line in cache_file.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            tasks.append(json.loads(line))

    return tasks[:n]


def extract_code_from_response(response: str) -> str:
    """Extract Python code from a model response.

    Handles:
    - Strips Qwen3.8 thinking tokens first
    - ```python ... ``` blocks
    - ```...``` blocks
    - Bare code (no fence)
    """
    # Strip thinking tokens first (Qwen3.8 generates reasoning before answers).
    response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    response = re.sub(r"<think>.*$", "", response, flags=re.DOTALL)

    # Try ```python ... ``` first.
    m = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try ```...``` (any language).
    m = re.search(r"```\w*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try to find function definitions.
    m = re.search(r"(def\s+\w+.*?)(?:\n\ndef|\Z)", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def run_humaneval_task(
    task: dict[str, Any],
    generated_code: str,
    timeout: int = 10,
) -> tuple[bool, str]:
    """Run a HumanEval task: combine generated code with test, execute.

    Returns (passed, error_message).
    """
    entry_point = task.get("entry_point", "")
    test_code = task.get("test", "")
    prompt = task.get("prompt", "")

    # The generated code should complete the function from the prompt.
    # HumanEval prompt ends mid-function; the model should complete it.
    full_code = prompt + generated_code + "\n\n" + test_code + "\n\n"
    full_code += f"check({entry_point})\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        env = dict(os.environ)
        env.pop("HTTP_PROXY", None)
        env.pop("HTTPS_PROXY", None)

        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout).strip()[:500]
    except subprocess.TimeoutExpired:
        return False, f"execution timed out after {timeout}s"
    except Exception as e:
        return False, f"execution error: {e}"
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


async def eval_humaneval(
    llm: Any,
    cfg: Any,
    n: int = 20,
    use_research: bool = True,
    search_cache: dict[str, list] | None = None,
) -> BenchmarkReport:
    """Run HumanEval benchmark.

    Args:
        llm: Model instance.
        cfg: UltRes config.
        n: Number of tasks to run.
        use_research: If True, use the research loop; if False, direct generation.
        search_cache: v1.6.1 Optional cache for search results across tasks.

    Returns:
        BenchmarkReport with pass@1 results.
    """
    cache_dir = Path(cfg.ultres_dir) / "eval_cache"
    tasks = download_humaneval(cache_dir, n)

    report = BenchmarkReport(benchmark="humaneval")
    start = time.time()

    for i, task in enumerate(tasks):
        task_id = task.get("task_id", f"task_{i}")
        prompt = task.get("prompt", "")

        if use_research:
            # Use the research loop to research the task, then generate code.
            from ultres.agent.research_loop import run_research_loop
            query = f"Write a Python function that: {prompt}"
            try:
                result = await run_research_loop(llm, cfg, query)
                generated = extract_code_from_response(result.answer)
                pages_researched = len(result.visited_urls)
            except Exception as e:
                generated = ""
                pages_researched = 0
                error = str(e)
        else:
            # Direct generation (no research).
            try:
                resp = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": (
                            "Complete the Python function. Output ONLY the function body "
                            "and any necessary imports. Do NOT include explanations, "
                            "comments about your reasoning, or the word 'thinking'. "
                            "Start your response directly with the code."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                )
                raw = resp["choices"][0]["message"]["content"]
                generated = extract_code_from_response(raw)
                pages_researched = 0
            except Exception as e:
                generated = ""
                pages_researched = 0

        # Run the test.
        passed, error = run_humaneval_task(task, generated)

        tr = TaskResult(
            task_id=task_id,
            prompt=prompt[:200],
            expected=task.get("canonical_solution", "")[:200],
            generated=generated[:500],
            passed=passed,
            error=error,
            elapsed=0.0,
            pages_researched=pages_researched,
        )
        report.results.append(tr)
        report.total_tasks += 1
        if passed:
            report.passed += 1
        else:
            report.failed += 1

        print(f"  [{i+1}/{n}] {task_id}: {'PASS' if passed else 'FAIL'}"
              + (f" — {error[:80]}" if error else ""))

    report.total_elapsed = time.time() - start
    if report.total_tasks > 0:
        report.avg_pages_researched = sum(
            r.pages_researched for r in report.results
        ) / report.total_tasks
    return report


# ---------------------------------------------------------------------------
# MMLU benchmark
# ---------------------------------------------------------------------------

MMLU_URL = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"


def download_mmlu(cache_dir: Path, n: int) -> list[dict[str, Any]]:
    """Download MMLU dataset (or use cached copy).

    Uses the OpenAI simple-evals single CSV file with all MMLU questions.
    Returns list of question dicts: {subject, question, choices, answer}.
    """
    import csv
    import random

    cache_file = cache_dir / "mmlu.csv"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not cache_file.exists():
        import httpx
        resp = httpx.get(MMLU_URL, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        cache_file.write_text(resp.text, encoding="utf-8")

    all_questions: list[dict[str, Any]] = []

    with cache_file.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_questions.append({
                "subject": row.get("Subject", "unknown"),
                "question": row.get("Question", ""),
                "choices": [
                    row.get("A", ""),
                    row.get("B", ""),
                    row.get("C", ""),
                    row.get("D", ""),
                ],
                "answer": row.get("Answer", "").strip().upper(),
            })

    # Shuffle and take n (reproducible).
    random.seed(42)
    random.shuffle(all_questions)
    return all_questions[:n]


def extract_mmlu_answer(response: str) -> str:
    """Extract A/B/C/D answer from model response."""
    # Strip thinking tokens first (Qwen3.8 generates reasoning before answers).
    response = re.sub(r"\n\s*", "", response, flags=re.DOTALL)
    response = re.sub(r"\n.*$", "", response, flags=re.DOTALL)
    # Look for "Answer: X" pattern.
    m = re.search(r"(?:answer|answer is|the answer is)\s*[:\s]*([ABCD])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Look for just a single letter at the start.
    m = re.match(r"\s*([ABCD])\b", response)
    if m:
        return m.group(1).upper()
    # Look for the last occurrence of A/B/C/D in the response.
    matches = re.findall(r"\b([ABCD])\b", response)
    if matches:
        return matches[-1].upper()
    return ""


async def eval_mmlu(
    llm: Any,
    cfg: Any,
    n: int = 50,
    use_research: bool = True,
    search_cache: dict[str, list] | None = None,
    model_mgr: Any = None,
    use_deep: bool = False,
) -> BenchmarkReport:
    """Run MMLU benchmark.

    Args:
        llm: Model instance (for fast mode / direct generation).
        cfg: UltRes config.
        n: Number of questions to run.
        use_research: If True, use research; if False, direct answering.
        search_cache: v1.6.1 Optional cache for search results across tasks.
        model_mgr: v1.6.1 ModelManager for deep mode (required if use_deep=True).
        use_deep: v1.6.1 If True, use the deep research pipeline (bulk crawl
            thousands of pages) instead of the fast agentic loop.

    Returns:
        BenchmarkReport with accuracy results.
    """
    cache_dir = Path(cfg.ultres_dir) / "eval_cache"
    questions = download_mmlu(cache_dir, n)

    report = BenchmarkReport(benchmark="mmlu")
    start = time.time()

    for i, q in enumerate(questions):
        question_text = q["question"]
        choices = q["choices"]
        expected = q["answer"]

        # Format as multiple choice.
        formatted = (
            f"Question: {question_text}\n\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n\n"
            f"Answer with ONLY the letter (A, B, C, or D)."
        )

        if use_research and use_deep and model_mgr is not None:
            # v1.6.1: Deep mode — bulk crawl thousands of pages, cluster,
            # summarize, compress, then answer from the master brief.
            from ultres.research.pipeline import run_deep_research
            from ultres.streaming import ResearchStreamer
            from rich.console import Console
            from ultres.agent.research_loop import _make_provider
            streamer = ResearchStreamer(console=Console(), enabled=True)
            # Create a search provider for this task.
            provider = _make_provider(cfg)
            query = f"Research this topic thoroughly and answer:\n\n{formatted}"
            try:
                result = await run_deep_research(
                    model_mgr=model_mgr,
                    cfg=cfg,
                    user_query=query,
                    provider=provider,
                    console=Console(),
                    streamer=streamer,
                )
                generated = result.answer
                pages_researched = result.pages_crawled
            except Exception as e:
                generated = f"Error: {e}"
                pages_researched = 0
        elif use_research:
            from ultres.agent.research_loop import run_research_loop
            query = f"Research and answer this multiple-choice question:\n\n{formatted}"
            try:
                result = await run_research_loop(llm, cfg, query)
                generated = result.answer
                pages_researched = len(result.visited_urls)
            except Exception as e:
                generated = ""
                pages_researched = 0
        else:
            try:
                resp = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "Answer the multiple-choice question. End with 'Answer: X'."},
                        {"role": "user", "content": formatted},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                generated = resp["choices"][0]["message"]["content"]
                pages_researched = 0
            except Exception:
                generated = ""
                pages_researched = 0

        predicted = extract_mmlu_answer(generated)
        passed = predicted == expected

        tr = TaskResult(
            task_id=f"{q['subject']}_{i}",
            prompt=question_text[:200],
            expected=expected,
            generated=predicted,
            passed=passed,
            error="" if passed else f"Expected {expected}, got {predicted}",
            elapsed=0.0,
            pages_researched=pages_researched,
        )
        report.results.append(tr)
        report.total_tasks += 1
        if passed:
            report.passed += 1
        else:
            report.failed += 1

        print(f"  [{i+1}/{len(questions)}] {tr.task_id}: {'PASS' if passed else 'FAIL'}"
              + (f" — {tr.error}" if tr.error else ""))

    report.total_elapsed = time.time() - start
    if report.total_tasks > 0:
        report.avg_pages_researched = sum(
            r.pages_researched for r in report.results
        ) / report.total_tasks
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the eval harness."""
    parser = argparse.ArgumentParser(description="UltRes benchmark evaluation")
    parser.add_argument(
        "--benchmark", choices=["humaneval", "mmlu"], required=True,
        help="Benchmark to run.",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of tasks to run.")
    parser.add_argument("--output", "-o", type=str, default="", help="Output JSON file.")
    parser.add_argument("--no-research", action="store_true", help="Skip research (direct generation).")
    parser.add_argument(
        "--deep", action="store_true",
        help="Use the deep research pipeline (bulk crawl thousands of pages). "
             "Much slower but far more thorough. Requires SearXNG running.",
    )
    parser.add_argument(
        "--max-pages", type=int, default=500,
        help="Max pages to crawl in deep mode (default 500). Use --max-pages 2000 for full research.",
    )
    args = parser.parse_args()

    # Load UltRes config + model.
    from ultres.config import UltResConfig
    from ultres.models.loader import ModelManager

    cfg = UltResConfig.load()
    cfg.ensure_dirs()

    # v1.6.1: Deep mode uses the deep pipeline (bulk crawl), fast mode uses the agentic loop.
    mode = "deep" if args.deep else "fast"
    if args.deep:
        cfg.deep_research.max_pages = args.max_pages
        print(f"Loading model ({cfg.model.selection}, deep mode: {cfg.deep_research.max_pages} pages)...")
    else:
        print(f"Loading model ({cfg.model.selection}, fast mode: 32K ctx, 55 GPU layers)...")
    mgr = ModelManager(cfg, mode=mode)
    llm = mgr.load_instruct()
    print(f"Model loaded. Running {args.benchmark} benchmark with n={args.n}...")

    use_research = not args.no_research

    # v1.6.1: Shared search cache across tasks (same queries = cached results).
    search_cache: dict[str, list] = {}

    if args.benchmark == "humaneval":
        report = asyncio.run(eval_humaneval(llm, cfg, n=args.n, use_research=use_research, search_cache=search_cache))
    else:
        report = asyncio.run(eval_mmlu(
            llm, cfg, n=args.n, use_research=use_research,
            search_cache=search_cache, model_mgr=mgr, use_deep=args.deep,
        ))

    # Print summary.
    print()
    print(f"{'=' * 60}")
    print(f"Benchmark: {report.benchmark}")
    print(f"Tasks: {report.total_tasks}")
    print(f"Passed: {report.passed}")
    print(f"Failed: {report.failed}")
    print(f"Score: {report.score:.2f}%")
    print(f"Total time: {report.total_elapsed:.1f}s")
    print(f"Avg pages researched: {report.avg_pages_researched:.1f}")
    print(f"{'=' * 60}")

    # Save report.
    report_dict = report.to_dict()
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        print(f"\nReport saved to {output_path}")
    else:
        # Default output location.
        default_path = Path(cfg.ultres_dir) / f"eval_{args.benchmark}_{int(time.time())}.json"
        default_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        print(f"\nReport saved to {default_path}")

    # Cleanup.
    try:
        mgr.unload_current()
    except Exception:
        pass


if __name__ == "__main__":
    main()
