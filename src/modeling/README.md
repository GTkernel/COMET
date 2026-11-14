# src/modeling

`lstm_autoencoder.py` defines the model. `train_model.py` delegates training to `pipeline.py`, which fits training-only preprocessing and uses lazy windows for calibration and scoring. `diagnostics.py` writes reconstruction-error reports for a recorded run.

Set training options in `config/model_configs/default.yaml` or pass another YAML file with `--config`.
