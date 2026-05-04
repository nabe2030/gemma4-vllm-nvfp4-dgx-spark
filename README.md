# Gemma 4 NVFP4 + vLLM on DGX Spark — Full Benchmark

Throughput, Japanese reasoning (JCQ), and VLM benchmarks for **Gemma 4 26B-A4B / 31B Dense** in NVFP4 quantization on **NVIDIA DGX Spark (GB10)**, comparing **vLLM** against **llama.cpp** under identical measurement conditions.

> **TL;DR**: As of May 2026, vLLM with NVFP4 finally works reliably on DGX Spark. The same Gemma 4 26B-A4B model runs **3.35× faster on vLLM than llama.cpp at concurrency 8**, while llama.cpp remains slightly faster at c=1. Gemma 4 31B Dense NVFP4 reaches **97.86% on JCQ** — tied for the best score we have measured on this hardware — but caps out at 6.92 tok/s due to memory bandwidth.

## Highlights

### vLLM finally works on DGX Spark

The day Gemma 4 launched (2026-04-02), NVIDIA published `nvidia/Gemma-4-31B-IT-NVFP4` and a DGX Spark-specific vLLM guide. Until late April however, vLLM crashed with `CUDA illegal instruction` the moment it tried to load NVFP4 weights on ARM64 + GB10 (SM 12.1). This was finally resolved when NVIDIA pushed the `vllm/vllm-openai:gemma4-cu130` image on **2026-04-10**, which fixed the NVFP4 GEMM path and the Gemma 4 MoE loader.

This repository captures benchmark results from a full validation session on 2026-05-02.

### Same model, different runtime — 3.35× difference at c=8

Running the **same Gemma 4 26B-A4B** under identical workload (200 ShareGPT prompts):

| Concurrency | vLLM (NVFP4) | llama.cpp (Q4_K_M) | Faster runtime | Margin |
|-------------|--------------|---------------------|----------------|--------|
| c=1  | 48.88 tok/s  | **61.98 tok/s**    | llama.cpp      | 1.27×  |
| c=8  | **207.48 tok/s** | 61.95 tok/s    | vLLM           | **3.35×** |
| c=32 | **441.14 tok/s** | 154.43 tok/s   | vLLM           | 2.86×  |

llama.cpp wins single-stream decode. vLLM dominates whenever continuous batching kicks in.

### Throughput sweep (vLLM, Gemma 4 26B-A4B NVFP4)

| Concurrency | Output tok/s | Peak tok/s | TPOT (ms) |
|------------:|-------------:|-----------:|----------:|
| 1  | 48.88  | 52  | 20.08 |
| 2  | 87.44  | 94  | 22.29 |
| 4  | 136.05 | 152 | 28.35 |
| 8  | 207.48 | 248 | 35.94 |
| 16 | 302.85 | 368 | 53.96 |
| 32 | 441.14 | 574 | 61.74 |

9.0× aggregate scaling from c=1 to c=32 with TPOT growing from 20 ms to 62 ms — textbook continuous-batching behavior.

### 31B Dense is bandwidth-bound

| Model | Active params | Theoretical limit (273 GB/s) | Measured c=1 | Achieved |
|-------|--------------:|-----------------------------:|-------------:|---------:|
| 31B Dense | 30.7B | 17.6 tok/s | 6.92 tok/s | 39.3% |
| 26B-A4B   | 4B    | 136.6 tok/s | 48.88 tok/s | 35.8% |

The 7.07× ratio between measured throughputs matches the 7.68× ratio of active parameters. On GB10's 273 GB/s LPDDR5X, single-stream Dense throughput is essentially fixed by `bandwidth / (active_params × bytes_per_param)`. Loading a 31B Dense model is technically possible because 128 GB unified memory holds it, but interactive use is impractical.

### JCQ (JCommonsenseQA v1.1, 1,119 questions, 3-shot)

| Model | Quant | Runtime | Thinking OFF | Thinking ON |
|-------|-------|---------|-------------:|------------:|
| Gemma 4 26B-A4B  | F16    | llama.cpp | 96.51% | bug (`<unused49>`) |
| **Gemma 4 26B-A4B**  | **NVFP4** | **vLLM** | **96.34%** | **93.48%** |
| **Gemma 4 31B Dense** | **NVFP4** | **vLLM** | **97.86%** | (10-sample 100%) |
| Qwen 3.5-35B-A3B | MXFP4  | llama.cpp | 96.16% | 89.28% |
| Qwen 3.6-35B-A3B | MXFP4  | llama.cpp | 95.98% | 96.43% |
| Nemotron 3 Super 120B | Q4_K | llama.cpp | 97.86% | 97.59% |

- 31B Dense NVFP4 ties Nemotron 3 Super 120B at 97.86% — top score in our 30B-class measurements.
- 26B-A4B shows only −0.17pt degradation from F16, confirming NVFP4 quantization preserves accuracy.
- The `<unused49>` flooding bug seen with F16 GGUF + llama.cpp is gone with NVFP4 + vLLM.
- 31B Dense Thinking ON would take ~11.6 hours for the full 1,119 questions, so we only ran a 10-question sample (100% correct). Not practical for interactive use.

### VLM: structured-output reliability went from 33% to 100%

| Model | Quant | Runtime | Caption (s/img) | JSON parse | PPE parse |
|-------|-------|---------|----------------:|-----------:|----------:|
| 26B-A4B  | F16    | llama.cpp | 40.14 | 40.0% | 33.3% |
| **26B-A4B**  | **NVFP4** | **vLLM** | **10.74** | **100%** | **100%** |
| 31B Dense | Q4_K_M | llama.cpp | 102.66 | 100% | 33.3% |
| **31B Dense** | **NVFP4** | **vLLM** | **72.27** | **100%** | **100%** |

- **MoE: 3.74× faster captioning** with vLLM + NVFP4 vs llama.cpp + F16
- **Dense: 1.42× faster** with vLLM + NVFP4 vs llama.cpp + Q4_K_M
- **PPE detection JSON parse rate: 33.3% → 100%** — the single biggest reliability improvement in this round of tests. Whatever combination of `--reasoning-parser gemma4`, NVFP4 quantization, and vLLM's serving path is responsible, structured-output workloads (RAG, agents, JSON extraction) are now in a different league.

## Methodology — `vllm bench serve` for both runtimes

Both vLLM and llama.cpp are driven by the **same client tool**: `vllm bench serve` (a.k.a. `benchmark_serving.py`, subcommand `serve`, vLLM 0.19.x), pointed at OpenAI-compatible `/v1/chat/completions` endpoints. llama.cpp's `llama-server` exposes the same endpoint, so the URL is the only difference.

```bash
# llama.cpp side, called from inside the vLLM container as a thin client:
docker run --rm -it --network host \
  -v ~/datasets:/datasets \
  -v ~/bench-results:/bench-results \
  -v ~/models/gemma4-26b-a4b-nvfp4:/tokenizer \
  --entrypoint vllm \
  vllm/vllm-openai:gemma4-cu130 \
    bench serve \
      --backend openai-chat \
      --base-url http://localhost:8080 \
      --endpoint /v1/chat/completions \
      --model gemma-4-26B-A4B-it-Q4_K_M.gguf \
      --tokenizer /tokenizer \
      --dataset-name sharegpt \
      --dataset-path /datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 200 \
      --max-concurrency 1 \
      --save-result \
      --result-dir /bench-results/llamacpp \
      --result-filename gemma4-26b-q4km-c1.json
```

This eliminates per-tool measurement bias (different random prompt selection, different tokenization, different client overhead) and lets numbers be compared directly.

Dataset: `ShareGPT_V3_unfiltered_cleaned_split.json` (642 MB, ~90,000 conversations). 200 prompts per run, sweeping `--max-concurrency` over `{1, 2, 4, 8, 16, 32}`.

## Server startup recipes

### vLLM — Gemma 4 31B Dense NVFP4

```bash
docker run -d --runtime nvidia --gpus all --shm-size=16g \
  -v ~/models/gemma4-31b-it-nvfp4:/models/gemma4-31b-it-nvfp4 \
  --name vllm-gemma4-31b -p 8000:8000 \
  vllm/vllm-openai:gemma4-cu130 \
    --model /models/gemma4-31b-it-nvfp4 \
    --quantization modelopt \
    --reasoning-parser gemma4 \
    --served-model-name gemma4-31b-nvfp4 \
    --max-model-len 8192
```

### vLLM — Gemma 4 26B-A4B NVFP4 (MoE, requires patch)

The official vLLM Gemma 4 MoE loader has a compatibility issue ([vLLM #38912](https://github.com/vllm-project/vllm/issues/38912)). The `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` model ships with a patched `gemma4_patched.py` that has to be bind-mounted over the in-container loader. `--moe-backend marlin` is also mandatory because GB10 (SM 12.1) lacks native FP4 compute and Marlin provides the W4A16 fallback path.

```bash
docker run -d --runtime nvidia --gpus all --shm-size=16g \
  -v ~/models/gemma4-26b-a4b-nvfp4:/models/gemma4-26b-a4b-nvfp4 \
  -v ~/models/gemma4-26b-a4b-nvfp4/gemma4_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py \
  --name vllm-gemma4-26b -p 8000:8000 \
  vllm/vllm-openai:gemma4-cu130 \
    --model /models/gemma4-26b-a4b-nvfp4 \
    --quantization modelopt \
    --kv-cache-dtype fp8 \
    --moe-backend marlin \
    --reasoning-parser gemma4 \
    --served-model-name gemma4-26b-nvfp4 \
    --max-model-len 8192
```

Verify in startup logs that you see **both**:

```
Using 'MARLIN' NvFp4 MoE backend
Using NvFp4LinearBackend.FLASHINFER_CUTLASS for NVFP4 GEMM
```

If `MARLIN` is not selected and `CUTLASS_FP4` is used instead, the model produces NaN garbage. A quick sanity test in Japanese (e.g. "日本の首都はどこですか") will reveal it immediately as garbled output.

### llama.cpp — Gemma 4 26B-A4B Q4_K_M

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma4-26b-a4b/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -fa on --jinja --reasoning off \
  --parallel 32 -c 262144
```

## Hardware

- NVIDIA DGX Spark (GB10 Grace Blackwell, SM 12.1)
- 128 GB LPDDR5X unified memory @ 273 GB/s
- Ubuntu (Linux 6.14) / CUDA 13.0 / Driver 580.x

## Repository layout

```
.
├── README.md                            # this file
├── results/
│   ├── throughput/                      # vllm bench serve outputs (9 files)
│   │   ├── gemma4-26b-nvfp4-c{1,2,4,8,16,32}.json
│   │   ├── gemma4-31b-nvfp4-c{1,8,32}.json
│   │   └── gemma4-26b-q4km-c{1,8,32}.json
│   ├── jcq/                             # JCommonsenseQA results (4 files)
│   │   ├── jcq_gemma4-26b-nvfp4_nothink.json
│   │   ├── jcq_gemma4-26b-nvfp4_think.json
│   │   ├── jcq_gemma4-31b-nvfp4_nothink.json
│   │   └── jcq_gemma4-31b-nvfp4_think_sample10.json
│   └── vlm/                             # VLM benchmark results (2 files)
│       ├── vlm_gemma4-26b-nvfp4.json
│       └── vlm_gemma4-31b-nvfp4.json
├── scripts/
│   ├── jcq_bench.py                     # JCQ runner (Thinking OFF, OpenAI-compatible API)
│   ├── jcq_bench_thinking.py            # JCQ runner with Thinking ON support
│   └── vlm_bench.py                     # Caption + JSON + PPE benchmark runner
└── charts/                              # 6 PNGs + matplotlib script
    ├── chart_throughput_vs_concurrency.png
    ├── chart_tpot_vs_concurrency.png
    ├── chart_jcq_accuracy.png
    ├── chart_vlm_caption_speed.png
    ├── chart_vlm_parse_rate.png
    ├── chart_bandwidth_bound.png
    └── make_charts_vllm.py
```

## Reproducing the benchmarks

1. **Pull the vLLM image** (used for both serving and as a benchmark client):
   ```bash
   docker pull vllm/vllm-openai:gemma4-cu130
   ```

2. **Download the ShareGPT dataset** (also serves as a benchmark client to llama.cpp):
   ```bash
   hf download anon8231489123/ShareGPT_Vicuna_unfiltered \
     ShareGPT_V3_unfiltered_cleaned_split.json \
     --local-dir ~/datasets/sharegpt
   ```

3. **Download the models** (32 GB + 16 GB):
   ```bash
   hf download nvidia/Gemma-4-31B-IT-NVFP4 --local-dir ~/models/gemma4-31b-it-nvfp4
   hf download bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4 --local-dir ~/models/gemma4-26b-a4b-nvfp4
   hf download ggml-org/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-Q4_K_M.gguf --local-dir ~/models/gemma4-26b-a4b
   ```

4. **Start one server at a time** using the recipes above, then run benchmarks via `vllm bench serve`.

5. **For JCQ and VLM**, use the scripts under `scripts/` — they target the OpenAI-compatible endpoint and work against either runtime by changing `--api-url`.

## Related

- **Full Japanese article on Qiita**: [DGX Spark で Gemma 4 26B-A4B / 31B Dense を vLLM でベンチマーク測定 ~ llama.cpp と比較](https://qiita.com/nabe2030/items/e0b19aeabe921f796000)
- **Previous repository (Gemma 4 vs Qwen 3.5 with llama.cpp)**: [nabe2030/gemma4-vs-qwen35-dgx-spark](https://github.com/nabe2030/gemma4-vs-qwen35-dgx-spark)
- **Independent verification of the same hardware**: [ai-muninn: Gemma 4 26B NVFP4 52 toks](https://ai-muninn.com/en/blog/dgx-spark-gemma4-26b-nvfp4-52-toks) (their 52 tok/s c=1 result matches our 48.88 tok/s)

## License

- Code in this repository: MIT
- Benchmark result JSONs: same license as the source datasets and models referenced
- Charts and figures: free to reuse with attribution
