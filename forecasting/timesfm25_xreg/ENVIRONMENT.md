# Environment for timesfm25_xreg (second foundation batch)

Python 3.11.17 venv inside the worktree; full freeze in `requirements.lock.txt`.

## Reproduce

```bash
cd /root/workspace/Code/DualTimesField-foundation-baselines
python3.11 -m venv .venvs/timesfm25
.venvs/timesfm25/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venvs/timesfm25/bin/pip install timesfm==2.0.2 pandas utilsforecast pyyaml
```

`timesfm` 2.0.2 does not declare torch as a dependency — install torch first
(cu126 index; the host driver 560.35.05 cannot run cu130 wheels).

## Weights

`google/timesfm-2.5-200m-pytorch` revision `1d952420fba8…`, Apache-2.0:

```bash
export HF_HOME=$PWD/weights/hf_home
.venvs/timesfm25/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('google/timesfm-2.5-200m-pytorch'))"
```

Workers run with `HF_HUB_OFFLINE=1`.
