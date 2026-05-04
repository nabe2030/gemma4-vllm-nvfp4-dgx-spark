# Commit Message Suggestions for `gemma4-vllm-nvfp4-dgx-spark`

## Initial commit (creating the repo)

### Short version (50 chars goal, 72 max for header)

```
Initial: Gemma 4 NVFP4 + vLLM benchmark on DGX Spark
```

### Conventional Commits style

```
feat: initial benchmark suite for Gemma 4 NVFP4 + vLLM on DGX Spark

- Throughput sweep (c=1..32) for 26B-A4B NVFP4 (vLLM),
  31B Dense NVFP4 (vLLM), and 26B-A4B Q4_K_M (llama.cpp)
- JCQ (1119 questions) for 26B-A4B and 31B Dense, both
  Thinking OFF and ON
- VLM (Caption / JSON extract / PPE detect) for both NVFP4 models
- All measurements driven by vllm bench serve for direct
  cross-runtime comparison
- Server startup recipes and reproduction guide in README
```

### Plain-English version

```
Initial commit: Gemma 4 NVFP4 vs llama.cpp Q4_K_M on DGX Spark

Adds benchmark results and reproduction recipes for running
Gemma 4 26B-A4B and 31B Dense in NVFP4 quantization with vLLM
on NVIDIA DGX Spark (GB10), validated 2026-05-02. Companion
to the Qiita article published on the same day.
```

## Subsequent commits (suggested patterns)

### Adding new benchmark results

```
results: add throughput sweep for <model> at c=<N>
results: add JCQ Thinking <ON|OFF> for <model>
results: add VLM benchmark for <model>
```

### Adding scripts

```
scripts: add <name>.py for <purpose>
scripts: extend jcq_bench.py with --thinking flag
```

### Updating charts

```
charts: regenerate after adding <model> data
charts: tweak annotations on figure <N>
```

### Updating README

```
docs: clarify <section> in README
docs: add reproduction step for <X>
docs: link to Qiita article
```

## PR description template (if you ever take PRs)

```markdown
## Summary

<one-line summary of the change>

## Changes

- ...
- ...

## Verification

- [ ] All existing benchmark JSONs still parse
- [ ] Charts regenerate without error
- [ ] README links resolve
```

## Tag suggestions

If you want to mark this initial dataset as a snapshot:

```
v2026-05-02-initial
```

Or if you prefer semver-style:

```
v0.1.0  # initial benchmark suite
```
