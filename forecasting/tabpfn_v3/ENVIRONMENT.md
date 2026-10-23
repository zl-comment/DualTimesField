# Environment for tabpfn_v3 (second foundation batch)

Python 3.11.17 venv created inside the worktree; see `requirements.lock.txt`
for the full frozen dependency list (`pip freeze`).

## Reproduce

```bash
cd /root/workspace/Code/DualTimesField-foundation-baselines
python3.11 -m venv .venvs/tabpfn_v3
.venvs/tabpfn_v3/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venvs/tabpfn_v3/bin/pip install tabpfn==9.2.0 pyyaml pandas
```

The NVIDIA driver on this host (560.35.05 / CUDA 12.6) cannot run the default
`cu130` wheels that PyPI resolves for torch 2.14.1; the cu126 index above is
required. Installing torch first prevents tabpfn's dependency resolver from
pulling the cu130 build.

## Weights

`Prior-Labs/tabpfn_3` revision `24a16a89d245878b846555110985634aa2e656d7`,
file `tabpfn-v3-regressor-v3_default.ckpt` (232,679,299 bytes; SHA256 in
`source.json`). Downloaded once from the public HuggingFace repo into the
worktree-level cache:

```bash
export HF_HOME=$PWD/weights/hf_home
.venvs/tabpfn_v3/bin/python -c "from huggingface_hub import hf_hub_download as d; print(d('Prior-Labs/tabpfn_3','tabpfn-v3-regressor-v3_default.ckpt'))"
```

The tabpfn package's own downloader presents an interactive license gate
(TabPFNLicenseError; account flow at https://ux.priorlabs.ai, `TABPFN_TOKEN`).
The adapter does not use that downloader and never sends data anywhere: it
points `model_path` at the already-downloaded local checkpoint file. The
checkpoint is licensed `tabpfn-3-license-v1.0` (research and internal
evaluation permitted; no commercial use). No account token is required for
this research use, and none is stored or logged. If institutional policy later
requires the official license flow, create an account at the URL above, accept
the TabPFN-3 license on the Licenses tab, and set `TABPFN_TOKEN` — the missing
items are exactly that account acceptance step and the token.

Workers run with `HF_HUB_OFFLINE=1`; no dependency or weight updates occur
during runs.
