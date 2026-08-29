# Reproducibility protocol

The reference experiment uses Python 3.10.8, PyTorch 2.1.2+cu118, CUDA 11.8,
cuDNN 8.7.0, NumPy 1.26.4, Pillow 10.3.0, and one RTX 4090. Install the exact
Python packages with:

```bash
python -m pip install -r requirements-repro.txt
python -m pip install -e .
```

Use the official `dinov2_vitb14_pretrain.pth` file. The reference SHA256 is:

```text
0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
```

Build UAV90K with the repository conversion script. It performs a per-city
85/5/10 shuffle with seed 42, producing 76,500 train, 4,500 validation, and
9,000 test queries. Verify the resulting metadata and split fingerprint before
training:

```bash
PYTHONPATH=src python scripts/fingerprint_dataset.py --dataset /path/to/UAV90K
```

The reference UAV90K metadata fingerprint is:

```text
fdaf3f74fed5e8fa953152a7d4c7bfbd02f0becdf1ff89a99dace52c3ad4001a
```

Training defaults to seed 42 and deterministic mode. Every run writes
`config.json` and `reproducibility.json`, and embeds the same manifest in each
new checkpoint. Evaluation refuses a checkpoint/dataset fingerprint mismatch.
Keep batch size, worker count, AMP mode, model version, and all command-line
arguments identical to the reference run.

Exact checkpoint bytes are expected only on an identical software and hardware
stack. A different CUDA, cuDNN, PyTorch, or GPU version can introduce small
floating-point changes. Report mean and standard deviation over three seeds
(42, 123, 3407) for publication-grade reproducibility; use seed 42 for the
single reference result.

The primary MLE protocol is `bearing-compatible`: same-city pixel errors are
converted with the map scale and macro-averaged across cities; cross-city
predictions remain LSR failures and are reported separately. The stricter
global number is available with `--mle-protocol global-geodesic`. Always state
the selected protocol with reported results.

The seed-42 reference test result is:

| Recall@1 | LSR@15 | HSR@15 | MLE | MHE |
|---:|---:|---:|---:|---:|
| 72.01% | 82.73% | 70.16% | 14.10 m | 14.31 deg |

On the identical RTX 4090 reference stack, deterministic reruns should be
effectively identical. On a different supported CUDA GPU/software stack,
differences within 0.5 percentage points for success rates and within 0.5 m or
0.5 degrees for mean errors are considered a successful reproduction. Larger
differences require checking the recorded dataset, DINO weight, code revision,
and runtime manifests before interpreting model variance.

For the exact two-stage command sequence, export `UAV90K_ROOT`,
`DINOV2_REPO`, and `DINOV2_WEIGHTS`, then run:

```bash
bash scripts/run_reference_pipeline.sh /path/to/fresh/output-directory
```
