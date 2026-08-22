"""Benchmark Qwen3.8-27B-UD-IQ2_XXS on RTX 4060 Ti 8GB."""
import os, sys, time, pathlib

# Fix DLL search path for CUDA runtime
nvidia_base = pathlib.Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
for sub in ["cuda_runtime/bin", "cublas/bin", "cuda_nvrtc/bin"]:
    p = nvidia_base / sub
    if p.exists():
        os.add_dll_directory(str(p))

from llama_cpp import Llama

MODEL = str(pathlib.Path.home() / ".ultres" / "models" / "Qwen3.8-27B-UD-IQ2_XXS.gguf")

print(f"Loading model: {MODEL}")
print("Config: -ngl 50, -c 65536, q4_0 KV cache, flash attn")
print()

t0 = time.time()
llm = Llama(
    model_path=MODEL,
    n_gpu_layers=50,
    n_ctx=65536,
    n_batch=512,
    flash_attn=True,
    type_k=2,  # q4_0
    type_v=2,  # q4_0
    verbose=True,
)
t_load = time.time() - t0
print(f"\nLoad time: {t_load:.1f}s")
print()

# Warmup
print("Warmup...")
llm.create_chat_completion(messages=[{"role": "user", "content": "Hi"}], max_tokens=8, temperature=0.1)

# Benchmark
prompt = "Write a Python function that checks if a string is a palindrome. Include docstring and type hints."
print(f"\nPrompt: {prompt}")
print("Generating...")

t0 = time.time()
resp = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=256,
    temperature=0.7,
    top_p=0.95,
    top_k=20,
    stream=True,
)
tokens = 0
for chunk in resp:
    delta = chunk["choices"][0].get("delta", {})
    if "content" in delta and delta["content"]:
        tokens += 1
        print(delta["content"], end="", flush=True)
t_gen = time.time() - t0

print(f"\n\n--- Results ---")
print(f"Tokens generated: {tokens}")
print(f"Generation time: {t_gen:.1f}s")
print(f"Tokens/sec: {tokens/t_gen:.1f}")
print(f"Load time: {t_load:.1f}s")
