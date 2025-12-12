# Model Configuration

The `model` section of your `config.yaml` tells `stimulus-py` how to instantiate your model and what hyperparameters to optimize.

## Basic Structure

```yaml
model:
  path: "path/to/my_model.py"     # Path to python file
  class_name: "MyCustomModel"     # Class name in that file
  
  # Static parameters are passed directly to __init__
  # These are constant across all trials
  static_params:
    input_dim: 50
    output_dim: 2000
    act_fn: "ReLU"
```

## Hyperparameter Search Space

`stimulus-py` shines when optimizing hyperparameters. You define "search spaces" in `network_params` and `optimizer_params`.

### Parameter Types

*   `categorical`: Choose one from a list.
*   `int` / `float`: Choose a value in a range.
*   `log: true`: Sample logarithmically (good for learning rates).

### Example

```yaml
# Passed to model.__init__
network_params:
  hidden_dim:
    type: "int"
    low: 64
    high: 1024
    step: 64
  
  dropout:
    type: "float"
    low: 0.0
    high: 0.5

# Passed to torch.optim.Optimizer
optimizer_params:
  method:
    type: "categorical"
    choices: ["Adam", "SGD", "RMSprop"]
  
  lr:
    type: "float"
    low: 1e-4
    high: 1e-1
    log: true
  
  weight_decay:
    type: "float"
    low: 0.0
    high: 1e-3
```

When `stimulus-py` runs a trial, it will:
1.  Sample a value for each parameter.
2.  Pass `network_params` samples to `MyCustomModel(**params)`.
3.  Pass `optimizer_params` samples to the optimizer constructor.
