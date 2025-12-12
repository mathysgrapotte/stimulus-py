# Dataset Configuration

In `stimulus-py`, data configuration happens in two distinct contexts:
1.  **Data Processing**: Configuring how raw data is encoded, split, and transformed (used by `stimulus split`, `stimulus transform`).
2.  **Training Integration**: Configuring how the model loads data during tuning/training.

## 1. Data Processing Config

When using the CLI tools to prepare your data, you use a YAML configuration that follows the `ConfigDict` schema. This file defines which columns to use, how to encode them, how to split the data, and any transformations to apply.

The schema consists of four main sections:

### Global Parameters
Global settings for the data processing pipeline.
```yaml
global_params:
  seed: 42
```

### Columns
Defines the input columns and their encoding.

```yaml
columns:
  - column_name: "gene_expression"
    column_type: "float"
    encoder:
      - name: "identity" # or specialized encoder
        params:
           dtype: "float32"
```

### Transforms
Defines transformations to apply to columns (e.g., normalization).

```yaml
transforms:
  - transformation_name: "log1p"
    columns:
      - column_name: "gene_expression"
        transformations:
          - name: "log1p"
            params: {}
```

### Split
Defines how to split the dataset.

```yaml
split:
  - split_method: "random"
    split_input_columns: ["sample_id"]
    params:
      ratios: [0.8, 0.1, 0.1] # Train, Val, Test
```

## 2. Training Integration

When running `stimulus tune` or `stimulus train`, you configure data loading in your main experiment config (the one with `model`, `objective`, etc.).

This involves two sections in the YAML:

### `dataset`
Use this section to specify the dataset class and path. This follows the `DatasetConfig` schema.

```yaml
dataset:
  type: "HuggingFaceDataset" # or "H5adDataset"
  params:
    path: "/path/to/processed_data" 
    # Additional dataset-specific loading args can go here
```

### `data_params`
Use this section for **tunable** data loader parameters, such as batch size. These are passed to the optimization engine and can be swept over.

```yaml
data_params:
  batch_size:
    mode: "int"
    params:
      low: 32
      high: 512
      step: 32
```
