# CLI Tooling

`stimulus-py` provides a suite of CLI tools to manage the entire lifecycle of your deep learning project.

## `stimulus tune`

This is the main workhorse for hyperparameter optimization.

```bash
stimulus tune --config config.yaml
```

**Key Features**:
*   Runs multiple trials (based on `n_trials` in config).
*   Uses Optuna TPE sampler by default.
*   Saves the best model to `output_dir`.
*   Logs all metrics to `sqlite` database or Journal file.

## `stimulus train`

Use this when you have a single configuration (no hyperparameter search) or want to retrain a specific set of parameters.

```bash
stimulus train --config config.yaml
```

Unlike `tune`, `train` runs exactly once.

## `stimulus split`

Helper utility to split a dataset into train/val/test partitions.

```bash
stimulus split --input my_data.parquet --output-dir my_split_data --ratios 0.8 0.1 0.1
```

## `stimulus transform`

Apply transformations (like normalization or feature selection) to your dataset *before* training. This is useful for expensive preprocessing steps.

```bash
stimulus transform --input raw_data --output proc_data --method log1p
```

## `stimulus predict`

Run inference using a trained model.

```bash
stimulus predict \
    --model-path output/experiment/best_model.pt \
    --data-path test_data \
    --output predictions.parquet
```
