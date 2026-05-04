# Update Snippet for `gemma4-vs-qwen35-dgx-spark` README

This is a small section to add to the **existing** `gemma4-vs-qwen35-dgx-spark`
README, pointing readers to the new `gemma4-vllm-nvfp4-dgx-spark` repository.

Recommended placement: under "Related" / "See also" section, or as a new top-level
section just below the "Conclusion".

---

## Suggested addition (English)

### Follow-up: vLLM + NVFP4 benchmark (May 2026)

A follow-up benchmark using **vLLM with NVFP4 quantization** on the same DGX Spark
hardware is available in a separate repository:

[**nabe2030/gemma4-vllm-nvfp4-dgx-spark**](https://github.com/nabe2030/gemma4-vllm-nvfp4-dgx-spark)

Highlights:

- vLLM finally works on DGX Spark (illegal-instruction issue resolved as of
  `vllm/vllm-openai:gemma4-cu130`, pushed 2026-04-10)
- Same Gemma 4 26B-A4B is **3.35× faster on vLLM than llama.cpp at concurrency 8**
- Gemma 4 31B Dense NVFP4 reaches **97.86% on JCQ** (top-tier in 30B class),
  but bandwidth-limited to 6.92 tok/s at c=1
- VLM structured-output reliability (PPE detect JSON parse) jumps from
  33.3% to 100% with vLLM + NVFP4

The new repository uses `vllm bench serve` (`benchmark_serving.py`) as the
unified measurement client for both vLLM and llama.cpp, replacing the
`llama-bench` based methodology used here.

The Japanese write-up is on Qiita:
[DGX Spark で Gemma 4 26B-A4B / 31B Dense を vLLM でベンチマーク測定 ~ llama.cpp と比較](https://qiita.com/nabe2030/items/e0b19aeabe921f796000)

---

## Alternative: Shorter version (one paragraph)

If the README is already long, a single-paragraph mention works too:

```markdown
**Update (May 2026)**: A follow-up benchmark with vLLM + NVFP4 on the same
DGX Spark hardware is in [nabe2030/gemma4-vllm-nvfp4-dgx-spark](https://github.com/nabe2030/gemma4-vllm-nvfp4-dgx-spark) — see also the
[Qiita article](https://qiita.com/nabe2030/items/e0b19aeabe921f796000).
The vLLM + NVFP4 stack is now stable and shows 3.35× higher throughput at
c=8 than llama.cpp + Q4_K_M for the same Gemma 4 26B-A4B model.
```

---

## Commit message for this update

```
docs: link to follow-up repo (gemma4-vllm-nvfp4-dgx-spark)
```
