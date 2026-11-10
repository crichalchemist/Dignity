# Results

The evaluation protocol and the numbers arrive in M6 and M7.

## Colab runbook

```bash
SHA=$(git rev-parse HEAD)        # must already be pushed to origin
SEED=tuning                      # M7 final run only: SEED=eval
mkdir -p runs/colab
colab new -s dignity-eval --gpu T4
colab exec -s dignity-eval -f scripts/colab_eval.py --timeout 43200 \
    --env DIGNITY_SHA="$SHA" --env DIGNITY_SEED="$SEED"
colab download -s dignity-eval /content/dignity_results_$SEED.tar.gz runs/colab/dignity_results_$SEED.tar.gz
colab stop -s dignity-eval
tar -xzf runs/colab/dignity_results_$SEED.tar.gz -C runs/colab
```

Tested with google-colab-cli 0.7.2. `--timeout` is required, because the default of 30 s kills the run.

The run executes `python -m eval.run --privacy-sweep` after the three tasks. It will write `privacy.json` beside the other results and `privacy_curve.png` beside the other charts, in the downloaded archive.

`risk.json` records `git_sha` and `config_hash`, so a reader can check which commit and config produced a result.
