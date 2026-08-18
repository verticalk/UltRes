"""Registry of UltRes-supported base models.

Each entry pins a HuggingFace GGUF repo + filename + sha256, the native context
window, and the recommended quantization. v1 only ships the "stock" entry;
v1.5/v2 add the custom UltRes checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    # HuggingFace repo id hosting the GGUF files.
    hf_repo: str
    # Filename within the repo (will be parameterized by quant).
    hf_filename_template: str
    # Native context window (tokens).
    native_ctx: int
    # Total params (billions) — for display + RAM heuristics.
    params_b: float
    # Default quantization.
    default_quant: str
    # Human-readable label.
    label: str
    # Whether this is a custom UltRes checkpoint (vs stock upstream).
    is_ultres: bool = False
    # sha256 of the default-quant GGUF file (filled in once pinned).
    sha256: str | None = None


_REGISTRY: dict[str, ModelSpec] = {
    # Default: general-purpose instruct model with native tool calling support.
    "stock": ModelSpec(
        key="stock",
        hf_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
        hf_filename_template="qwen2.5-7b-instruct-{quant_lower}.gguf",
        native_ctx=32_768,
        params_b=7.0,
        default_quant="Q4_K_M",
        label="Qwen2.5-7B-Instruct (stock)",
        is_ultres=False,
        # sha256 pinned once the exact file is confirmed at install time.
    ),
    # Code-specific tasks: coding model for when the query is purely about code.
    "coder": ModelSpec(
        key="coder",
        hf_repo="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        hf_filename_template="qwen2.5-coder-7b-instruct-{quant_lower}.gguf",
        native_ctx=32_768,
        params_b=7.0,
        default_quant="Q4_K_M",
        label="Qwen2.5-Coder-7B-Instruct (code-focused)",
        is_ultres=False,
    ),
    # v1.5: populated when UltRes-Base-7B is published.
    "ultres-base": ModelSpec(
        key="ultres-base",
        hf_repo="ultres/UltRes-Base-7B-GGUF",
        hf_filename_template="ultres-base-7b-{quant_lower}.gguf",
        native_ctx=32_768,
        params_b=7.0,
        default_quant="Q4_K_M",
        label="UltRes-Base-7B (QLoRA-specialized)",
        is_ultres=True,
    ),
    # v2: populated when UltRes-Base-7B-Long is published.
    "ultres-base-long": ModelSpec(
        key="ultres-base-long",
        hf_repo="ultres/UltRes-Base-7B-Long-GGUF",
        hf_filename_template="ultres-base-7b-long-{quant_lower}.gguf",
        native_ctx=262_144,
        params_b=7.0,
        default_quant="Q4_K_M",
        label="UltRes-Base-7B-Long (YaRN 256K)",
        is_ultres=True,
    ),
}


def get_spec(key: str) -> ModelSpec:
    """Return the ModelSpec for a registered key."""
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{key}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def list_specs() -> dict[str, ModelSpec]:
    """Return the full registry."""
    return dict(_REGISTRY)


def filename_for(spec: ModelSpec, quant: str | None = None) -> str:
    """Render the GGUF filename for a given quantization.

    Supports both {quant} (as-is) and {quant_lower} (lowercased) templates,
    since different HuggingFace repos use different casing conventions.
    """
    q = quant or spec.default_quant
    return spec.hf_filename_template.format(quant=q, quant_lower=q.lower())
