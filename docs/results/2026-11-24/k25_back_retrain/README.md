# Back retraining launch evidence (in progress)

This is a launch handoff, not a qualification result. One fresh Back-only continuation from the same K19 three-old parent, budget24000, GPU0. There is no automatic rollout evaluation or automatic follow-on recovery experiment. The user will return when training finishes.

`config.json` fixes training, source parent, future development criteria and checkpoint schedule. `train_back_qualified.py` imports archived K19 loading helpers to preserve the original trainable set, frozen base and normalizers. `back_training_protocol.py` implements deterministic episode-uniform whole/tail sampling and continuous warmup/cosine decay. `upload_manifest.json` binds exact uploaded source/config/start files. `LAUNCHED.json`, `startup_status.json`, `startup_console.log`, `training_manifest.json` and `first100_train.jsonl` verify a live detached process actually entered training; they are frozen STARTUP snapshots, not current/final output. Full logs, weights and optimizer states remain on the server. Do not claim qualification from training completion.

Validation from repository root:

```bash
python -m unittest discover -s tests -p test_back_training_protocol.py -v
python -m unittest discover -s tests -p test_stage1_protocol.py -v
```

Only launch/config/code and small startup evidence are included in Git; no checkpoint, dataset, cache or full future training output is uploaded. Refer to `docs/K25-Back-retrain.md` and the server's final completion/FAILED markers when the user returns.
