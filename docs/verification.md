# Phase 1 verification

Verified locally on Windows with Python 3.12.14 on 5 October 2026.

## Real model run

The pinned upstream model loaded successfully and predicted relationships for the included 1280 x 907 horse photograph and its two manual regions. This used the real RelateAnything model on CPU, with no mock fallback.

| Subject | Relation | Object | Model score |
| --- | --- | --- | --- |
| person-1 | riding | horse-1 | 0.551749 |
| horse-1 | carrying a rider | person-1 | 0.019899 |

The low-scoring second result is retained because the smoke-test score filter is zero. It should not be treated as a confirmed detection. These are model scores for one image and the configured vocabulary, not measured accuracy or calibrated CCTV confidence.

The first run took 75.82 seconds to download/load/initialize the model and 2.23 seconds for prediction. These are single-run wall-clock measurements, not steady-state FPS or a GPU benchmark.

The unmodified JSON result is saved in [phase1-sample-result.json](phase1-sample-result.json). Re-run using the commands in the README; newly generated results go to ignored `outputs/`.

## Checks

- Lightweight tests: 19 passed; real-model test excluded by default.
- Ruff lint: passed.
- Dependency consistency (`pip check`): passed.
- Real command-line inference: passed on CPU.
- Offline full suite with RUN_MODEL_TESTS=1, HF_HUB_OFFLINE=1, and TRANSFORMERS_OFFLINE=1: 20 passed in 17.65 seconds, including the real model integration test.

Exact installed package versions are recorded in [verified-environment-windows-py312.txt](verified-environment-windows-py312.txt). This is an environment snapshot for this Windows/Python combination, not a cross-platform lockfile.

## Scope and remaining work

The original repository contained only `# Work-mode` in its README. Phase 1 adds package structure, a pinned upstream adapter, strict input/configuration validation, a sample image and credits, JSON output, unit/integration tests, and lightweight Windows/Linux CI configuration. Synced project reference files were not changed.

CUDA and Linux execution were not tested locally. CI configuration is prepared but has not been run on GitHub. Automatic detection, video/camera input, tracking, temporal aggregation, event reasoning, and accuracy evaluation remain future phases described in [architecture.md](architecture.md).
