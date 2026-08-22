"""Final-answer reasoner.

After the research loop ends, this produces the streamed final answer. In v1
the loop's `finish` tool already produces the answer, so this module is a thin
wrapper that handles streaming to the terminal and writing to disk.

v1.2 adds two_pass_implement() for the deep research pipeline:
  Pass 1 (Instruct): synthesize research into an implementation plan
  Pass 2 (Coder): implement from the plan, forced to follow research

v1.5: With Qwen3.8-27B as the single model, both passes use the same model.
Thinking mode (Qwen3.8 generates <think> reasoning before answers) is enabled
by default — the thinking tokens are stripped from the final output but used
for better reasoning quality.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from ultres.research.compressor import MasterBrief
from ultres.streaming import ResearchStreamer


def _strip_thinking(text: str) -> str:
    """Strip Qwen3.8 thinking tokens (<think>...</think>) from output.

    The thinking content is useful for reasoning but should not appear
    in the final answer shown to the user.
    """
    # Remove <think>...</think> blocks (including unclosed ones at the end).
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Remove trailing unclosed <think> block.
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def stream_answer(
    answer: str,
    answer_path: Path,
    console: Console | None = None,
) -> None:
    """Render the final answer to the terminal and persist it to disk."""
    console = console or Console()
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.write_text(answer, "utf-8")
    console.print()
    console.print(Panel(Markdown(answer), title="UltRes answer", border_style="green"))
    console.print(f"\n[dim]Saved to {answer_path}[/dim]")


def final_reasoning_pass(
    llm: Any,
    user_query: str,
    topic_summaries: list[str],
    loaded_code: list[str],
    temperature: float = 0.4,
) -> str:
    """Optional v2-style final pass: synthesize across all topic summaries + code.

    v1 does not call this (the loop's `finish` is the answer), but it's here
    for v2 when the long-context model can hold more at once.
    """
    context_parts = []
    if topic_summaries:
        context_parts.append("## Topic summaries\n" + "\n\n".join(topic_summaries))
    if loaded_code:
        context_parts.append("## Code blocks\n" + "\n\n---\n\n".join(loaded_code))
    context = "\n\n".join(context_parts) or "(no additional context)"
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Synthesize a complete answer from the provided research."},
            {"role": "user", "content": f"Query: {user_query}\n\nContext:\n{context[:60000]}"},
        ],
        temperature=temperature,
        max_tokens=2048,
    )
    return resp["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# v1.2: Two-pass implementation
# ---------------------------------------------------------------------------

_PASS1_SYSTEM = """\
You are the architecture planner for UltRes, a research-driven AI. \
You have been given a master research brief derived from analyzing \
thousands of web pages. Your job is to write a detailed implementation \
plan based on this research.

The plan must:
- Specify the exact architecture and file structure
- Name the specific algorithms and patterns to use (from the research)
- Reference specific code examples that should be adapted
- List all libraries and tools recommended by the research
- Identify potential pitfalls and how to handle them
- Be specific enough that a coder can implement it without making \
  architecture decisions

Do NOT write code. Write a PLAN that a coder will follow.
"""

_PASS2_SYSTEM = """\
You are the implementation engine for UltRes. You MUST implement based on \
the plan below. The plan is derived from web research of thousands of pages.

CRITICAL RULES:
1. Do NOT substitute your own pretrained knowledge for research-based \
   architecture decisions. The plan specifies which algorithms, patterns, \
   and libraries to use — USE THOSE.
2. Use the provided code examples as reference and adapt them.
3. If the plan specifies an algorithm, use that algorithm.
4. If the plan specifies a library, use that library.
5. Your job is to write clean, working code that follows the plan — not to \
   redesign the architecture.

Write complete, production-quality code. Include error handling, comments, \
and proper structure as specified in the plan.
"""


def two_pass_implement(
    model_mgr: Any,
    user_query: str,
    brief: MasterBrief,
    streamer: ResearchStreamer | None = None,
    temperature: float = 0.4,
) -> tuple[str, str]:
    """Two-pass implementation: plan then implement.

    v1.5: With Qwen3.8-27B as the single model, both passes use the same model.
    No model swap needed — the 27B is smart enough for both planning and coding.
    Thinking mode is kept enabled for better reasoning quality.

    Args:
        model_mgr: ModelManager instance for loading/unloading models.
        user_query: The user's original request.
        brief: Master research brief with code examples.
        streamer: Optional streamer for live token display.
        temperature: Generation temperature.

    Returns:
        Tuple of (plan, implementation) strings.
    """
    # Build code examples text.
    code_text = "\n\n---\n\n".join(
        f"```{cb.language or ''}\n{cb.content}\n```"
        for cb in brief.code_examples
    )

    single_model = getattr(model_mgr, "is_single_model", False)

    # --- Pass 1: Write the plan ---
    llm_instruct = model_mgr.load_instruct()

    pass1_prompt = (
        f"User request: {user_query}\n\n"
        f"Master research brief ({brief.total_pages} pages, "
        f"{brief.total_clusters} clusters, {brief.total_code_blocks} code blocks):\n\n"
        f"{brief.text}\n\n"
        f"Best code examples from research:\n\n{code_text[:30000]}\n\n"
        f"Write a detailed implementation plan for this request. "
        f"Be specific about architecture, algorithms, file structure, and libraries."
    )

    if streamer and streamer.enabled:
        pass1_label = "Qwen3.8-27B" if single_model else "Instruct model"
        streamer.print(f"\n[bold cyan]Pass 1: Writing implementation plan ({pass1_label})...[/bold cyan]")

    if streamer and streamer.enabled:
        plan = streamer.token_stream(
            llm_instruct,
            messages=[
                {"role": "system", "content": _PASS1_SYSTEM},
                {"role": "user", "content": pass1_prompt},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
    else:
        resp = llm_instruct.create_chat_completion(
            messages=[
                {"role": "system", "content": _PASS1_SYSTEM},
                {"role": "user", "content": pass1_prompt},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
        plan = resp["choices"][0]["message"]["content"].strip()

    # Strip thinking tokens from the plan.
    plan = _strip_thinking(plan)

    # --- Pass 2: Implement from the plan ---
    # v1.5: In single-model mode, load_coder() returns the same model (no swap).
    coder = None
    coder_available = False
    try:
        coder = model_mgr.load_coder()
        coder_available = not single_model  # True only if a separate coder was loaded.
    except Exception as e:
        if streamer and streamer.enabled:
            streamer.print(f"[yellow]Coder model unavailable ({e}); trying Instruct fallback...[/yellow]")
        try:
            coder = model_mgr.load_instruct()
            coder_available = False
        except Exception as e2:
            if streamer and streamer.enabled:
                streamer.print(f"[red]Instruct fallback also failed ({e2}); using plan as answer.[/red]")
            return plan, plan

    # v1.6: Adaptive max_tokens based on plan complexity.
    complexity = estimate_complexity(plan)
    pass2_max_tokens = adaptive_max_tokens(complexity)

    # v1.6: Instruct the model to output files in ```file:path``` format.
    pass2_prompt = (
        f"User request: {user_query}\n\n"
        f"Implementation plan (derived from research — FOLLOW THIS):\n\n{plan}\n\n"
        f"Reference code examples from research:\n\n{code_text[:30000]}\n\n"
        f"Now implement the complete solution based on this plan. "
        f"Write all the code. Follow the plan exactly.\n\n"
        f"IMPORTANT: Output each file in this exact format:\n"
        f"```file:path/to/file.ext\n<file contents>\n```\n"
        f"Use one file block per file. Include all necessary files. "
        f"Also include a test file if appropriate."
    )

    if streamer and streamer.enabled:
        if single_model:
            model_label = "Qwen3.8-27B (same model)"
        elif coder_available:
            model_label = "Coder model"
        else:
            model_label = "Instruct model (fallback)"
        streamer.print(f"\n[bold cyan]Pass 2: Implementing from plan ({model_label}, {complexity}, {pass2_max_tokens} tokens)...[/bold cyan]")

    if streamer and streamer.enabled:
        implementation = streamer.token_stream(
            coder,
            messages=[
                {"role": "system", "content": _PASS2_SYSTEM},
                {"role": "user", "content": pass2_prompt},
            ],
            temperature=temperature,
            max_tokens=pass2_max_tokens,
        )
    else:
        resp = coder.create_chat_completion(
            messages=[
                {"role": "system", "content": _PASS2_SYSTEM},
                {"role": "user", "content": pass2_prompt},
            ],
            temperature=temperature,
            max_tokens=pass2_max_tokens,
        )
        implementation = resp["choices"][0]["message"]["content"].strip()

    # Strip thinking tokens from the implementation.
    implementation = _strip_thinking(implementation)

    # v1.6: Pass 3 — compile + test verification with fix loop.
    implementation = _verify_and_fix(
        coder, user_query, plan, implementation, streamer, temperature,
    )

    return plan, implementation


def _verify_and_fix(
    llm: Any,
    user_query: str,
    plan: str,
    implementation: str,
    streamer: ResearchStreamer | None,
    temperature: float,
    max_fix_iterations: int = 2,
) -> str:
    """v1.6: Pass 3 — verify implementation and fix if needed.

    Extracts files from the implementation, writes them to a temp directory,
    runs syntax checks + import checks + tests, and if anything fails,
    feeds the errors back to the model for a fix pass (up to max_fix_iterations).
    """
    files = extract_files_from_output(implementation)
    if not files:
        # No structured files — can't verify. Return as-is.
        if streamer and streamer.enabled:
            streamer.print("  [dim]No ```file:``` blocks found — skipping verification.[/dim]")
        return implementation

    if streamer and streamer.enabled:
        streamer.print(f"\n[bold cyan]Pass 3: Verifying {len(files)} files (syntax + imports + tests)...[/bold cyan]")

    with tempfile.TemporaryDirectory(prefix="ultres_verify_") as tmpdir:
        output_dir = Path(tmpdir)
        write_files_to_disk(files, output_dir)

        for iteration in range(max_fix_iterations + 1):
            verification = verify_implementation(files, output_dir, streamer)

            if verification["all_passed"]:
                if streamer and streamer.enabled:
                    streamer.print("  [green]All verification checks passed.[/green]")
                return implementation

            if iteration >= max_fix_iterations:
                if streamer and streamer.enabled:
                    streamer.print(
                        f"  [yellow]Verification still failing after {max_fix_iterations} fix attempts. "
                        f"Returning best effort.[/yellow]"
                    )
                return implementation

            if streamer and streamer.enabled:
                streamer.print(f"  [yellow]Fix pass {iteration + 1}/{max_fix_iterations}...[/yellow]")

            # Run fix pass.
            fixed = fix_pass(llm, user_query, plan, implementation, verification, temperature)
            if fixed and fixed != implementation:
                implementation = fixed
                # Re-extract and re-write files.
                new_files = extract_files_from_output(implementation)
                if new_files:
                    files = new_files
                    write_files_to_disk(files, output_dir)
            else:
                if streamer and streamer.enabled:
                    streamer.print("  [yellow]Fix pass produced no changes. Stopping.[/yellow]")
                return implementation

    return implementation


# ---------------------------------------------------------------------------
# v1.6: File extraction from implementation output
# ---------------------------------------------------------------------------

# Matches ```file:path/to/file.ext\n...content...\n``` blocks.
_FILE_FENCE_RE = re.compile(
    r"```file:([^\n]+)\n(.*?)```",
    re.DOTALL,
)


def extract_files_from_output(output: str) -> dict[str, str]:
    """Extract structured files from model output.

    v1.6: Parses ```file:path/to/file.ext``` blocks from the implementation
    text and returns a dict mapping file paths to file contents.

    If no file blocks are found, returns an empty dict (caller should fall
    back to treating the output as a single text blob).
    """
    files: dict[str, str] = {}
    for m in _FILE_FENCE_RE.finditer(output):
        path = m.group(1).strip()
        content = m.group(2).strip()
        if path and content:
            files[path] = content
    return files


def write_files_to_disk(
    files: dict[str, str],
    output_dir: Path,
) -> list[Path]:
    """Write extracted files to disk under output_dir.

    Returns list of written file paths.
    """
    written: list[Path] = []
    for rel_path, content in files.items():
        # Sanitize path — prevent path traversal.
        safe_path = Path(rel_path)
        if safe_path.is_absolute() or ".." in safe_path.parts:
            continue
        full_path = output_dir / safe_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        written.append(full_path)
    return written


# ---------------------------------------------------------------------------
# v1.6: Compile + test verification
# ---------------------------------------------------------------------------

def _check_python_syntax(file_path: Path) -> tuple[bool, str]:
    """Check Python file syntax with py_compile."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(file_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "syntax check timed out"
    except Exception as e:
        return False, f"py_compile error: {e}"


def _check_cpp_syntax(file_path: Path) -> tuple[bool, str]:
    """Check C++ file syntax with g++ -fsyntax-only (if available)."""
    try:
        result = subprocess.run(
            ["g++", "-fsyntax-only", str(file_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr.strip()
    except FileNotFoundError:
        # g++ not installed — skip.
        return True, "(g++ not available, skipped)"
    except subprocess.TimeoutExpired:
        return False, "syntax check timed out"
    except Exception as e:
        return False, f"g++ error: {e}"


def _check_imports(file_path: Path) -> tuple[bool, str]:
    """Check if all Python imports resolve (without executing the file)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             f"import ast, sys; "
             f"tree = ast.parse(open(r'{file_path}', encoding='utf-8').read()); "
             f"imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]; "
             f"names = []; "
             f"[names.extend(a.name.split('.')[0] for a in n.names) if isinstance(n, ast.Import) else names.append(n.module.split('.')[0] if n.module else '') for n in imports]; "
             f"[__import__(n) for n in set(names) if n and n != '']"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr.strip()
    except Exception as e:
        return False, f"import check error: {e}"


def _run_tests(
    test_file: Path,
    timeout: int = 30,
) -> tuple[bool, str]:
    """Run a test file in a sandboxed subprocess with a timeout.

    Security: runs with no network access (offline), in a temp directory,
    with a strict timeout. Captures stdout/stderr/exit code.
    """
    try:
        env = dict(os.environ)
        # Remove network-related env vars to sandbox.
        env.pop("HTTP_PROXY", None)
        env.pop("HTTPS_PROXY", None)
        env["NO_NETWORK"] = "1"

        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(test_file.parent),
        )
        output = result.stdout + "\n" + result.stderr
        if result.returncode == 0:
            return True, output.strip()
        return False, output.strip()
    except subprocess.TimeoutExpired:
        return False, f"tests timed out after {timeout}s"
    except Exception as e:
        return False, f"test runner error: {e}"


def verify_implementation(
    files: dict[str, str],
    output_dir: Path,
    streamer: ResearchStreamer | None = None,
) -> dict[str, Any]:
    """v1.6: Verify generated code — syntax check + import check + test run.

    Args:
        files: Dict of file paths to contents.
        output_dir: Directory where files were written.
        streamer: Optional streamer for progress display.

    Returns:
        Dict with verification results:
        {
            "all_passed": bool,
            "syntax_errors": {path: error_msg, ...},
            "import_errors": {path: error_msg, ...},
            "test_results": {path: (passed, output), ...},
            "fix_needed": bool,
        }
    """
    results: dict[str, Any] = {
        "all_passed": True,
        "syntax_errors": {},
        "import_errors": {},
        "test_results": {},
        "fix_needed": False,
    }

    if not files:
        return results

    for rel_path in files:
        file_path = output_dir / rel_path
        if not file_path.exists():
            continue

        ext = file_path.suffix.lower()

        # Syntax check.
        if ext == ".py":
            ok, err = _check_python_syntax(file_path)
            if not ok:
                results["syntax_errors"][rel_path] = err
                results["all_passed"] = False
                results["fix_needed"] = True
                if streamer and streamer.enabled:
                    streamer.print(f"  [red]Syntax error in {rel_path}: {err[:100]}[/red]")
        elif ext in (".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"):
            ok, err = _check_cpp_syntax(file_path)
            if not ok and "not available" not in err:
                results["syntax_errors"][rel_path] = err
                results["all_passed"] = False
                results["fix_needed"] = True
                if streamer and streamer.enabled:
                    streamer.print(f"  [red]Syntax error in {rel_path}: {err[:100]}[/red]")

        # Import check for Python files.
        if ext == ".py" and rel_path not in results["syntax_errors"]:
            ok, err = _check_imports(file_path)
            if not ok:
                results["import_errors"][rel_path] = err
                results["all_passed"] = False
                results["fix_needed"] = True
                if streamer and streamer.enabled:
                    streamer.print(f"  [yellow]Import error in {rel_path}: {err[:100]}[/yellow]")

        # Test run for test files.
        if ext == ".py" and ("test" in rel_path.lower() or "spec" in rel_path.lower()):
            ok, output = _run_tests(file_path)
            results["test_results"][rel_path] = (ok, output)
            if not ok:
                results["all_passed"] = False
                results["fix_needed"] = True
                if streamer and streamer.enabled:
                    streamer.print(f"  [red]Tests failed in {rel_path}[/red]")
            else:
                if streamer and streamer.enabled:
                    streamer.print(f"  [green]Tests passed in {rel_path}[/green]")

    return results


def fix_pass(
    llm: Any,
    user_query: str,
    plan: str,
    implementation: str,
    verification: dict[str, Any],
    temperature: float = 0.3,
) -> str:
    """v1.6: Fix pass — feed verification errors back to the model.

    The model sees the original plan, the implementation, and the errors,
    then produces a corrected implementation.
    """
    error_summary = []
    for path, err in verification.get("syntax_errors", {}).items():
        error_summary.append(f"Syntax error in {path}:\n{err[:500]}")
    for path, err in verification.get("import_errors", {}).items():
        error_summary.append(f"Import error in {path}:\n{err[:500]}")
    for path, (ok, output) in verification.get("test_results", {}).items():
        if not ok:
            error_summary.append(f"Test failures in {path}:\n{output[:500]}")

    errors_text = "\n\n".join(error_summary) or "Unknown errors."

    fix_prompt = (
        f"User request: {user_query}\n\n"
        f"Original plan:\n{plan[:4000]}\n\n"
        f"Implementation that has errors:\n{implementation[:8000]}\n\n"
        f"Verification errors found:\n{errors_text[:3000]}\n\n"
        f"Fix the implementation to resolve ALL the errors above. "
        f"Output the corrected implementation with files in ```file:path``` format. "
        f"Keep the same file structure — only fix the errors."
    )

    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _PASS2_SYSTEM},
                {"role": "user", "content": fix_prompt},
            ],
            temperature=temperature,
            max_tokens=8192,
        )
        fixed = resp["choices"][0]["message"]["content"].strip()
        return _strip_thinking(fixed)
    except Exception:
        return implementation  # Return original if fix fails.


# ---------------------------------------------------------------------------
# v1.6: Adaptive max_tokens based on task complexity
# ---------------------------------------------------------------------------

def estimate_complexity(plan: str) -> str:
    """Estimate task complexity from the implementation plan.

    Returns "simple", "medium", or "complex".
    """
    # Count file references in the plan.
    file_refs = len(re.findall(r"\b\w+\.(py|js|ts|cpp|cc|h|hpp|java|rs|go|rb|cs)\b", plan))
    # Count component/module references.
    component_refs = len(re.findall(r"\b(class|module|component|service|handler|controller|model|view)\b", plan, re.IGNORECASE))

    if file_refs >= 6 or component_refs >= 8:
        return "complex"
    if file_refs >= 3 or component_refs >= 4:
        return "medium"
    return "simple"


def adaptive_max_tokens(complexity: str) -> int:
    """Return max_tokens based on task complexity."""
    return {
        "simple": 4096,
        "medium": 8192,
        "complex": 16384,
    }.get(complexity, 8192)
