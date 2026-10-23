# Environment for moirai2_zeroshot (second foundation batch)

Python 3.11.17 venv inside the worktree; full freeze in `requirements.lock.txt`.

## Reproduce

The PyPI `uni2ts` 2.0.0 wheel predates the Moirai-2 classes, so the official
repository at a pinned commit is the implementation (installed `--no-deps`
after the PyPI install resolved the compatible dependency set; torch pinned by
uni2ts to 2.4.1, which runs on the host driver via cu121 wheels):

```bash
cd /root/workspace/Code/DualTimesField-foundation-baselines
python3.11 -m venv .venvs/moirai2
.venvs/moirai2/bin/pip install uni2ts==2.0.0 pyyaml pandas
git clone https://github.com/SalesforceAIResearch/uni2ts.git /tmp/uni2ts_src
git -C /tmp/uni2ts_src checkout cfd46d4510ed8896f263116f32928eede05b0a75
.venvs/moirai2/bin/pip install --no-deps --force-reinstall /tmp/uni2ts_src
```

Provenance: repository https://github.com/SalesforceAIResearch/uni2ts, commit
`cfd46d4510ed8896f263116f32928eede05b0a75`, Apache-2.0 (uni2ts code). Nothing
is vendored into this directory; the pinned commit IS the implementation and
is recorded in source.json and every artifact identity.

## Weights

`Salesforce/moirai-2.0-R-small` revision `30f43ff08c84…`, cc-by-nc-4.0
(research, non-commercial):

```bash
export HF_HOME=$PWD/weights/hf_home
.venvs/moirai2/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('Salesforce/moirai-2.0-R-small'))"
```

Workers run with `HF_HUB_OFFLINE=1`.
