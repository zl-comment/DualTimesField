# Environment for chronos2_zeroshot (second foundation batch)

Python 3.11.17 venv inside the worktree; full freeze in `requirements.lock.txt`.

## Reproduce

```bash
cd /root/workspace/Code/DualTimesField-foundation-baselines
python3.11 -m venv .venvs/chronos2
.venvs/chronos2/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venvs/chronos2/bin/pip install chronos-forecasting==2.3.2 "pandas[pyarrow]" pyyaml
```

Install torch FIRST from the cu126 index: the host driver (560.35.05) cannot
run the cu130 wheel PyPI resolves for torch 2.14.1 by default.

## Weights

`amazon/chronos-2` revision `29ec3766d36d…`, Apache-2.0, downloaded once:

```bash
export HF_HOME=$PWD/weights/hf_home
.venvs/chronos2/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('amazon/chronos-2'))"
```

Workers run with `HF_HUB_OFFLINE=1`; runs never update weights or deps.
