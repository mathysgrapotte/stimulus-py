# Model

The core of your research is the **Model**. `stimulus-py` uses a structural typing approach: your model class just needs to implement specific methods to work with the engine.

## The Interface

Your Python class needs to implement the following methods. It's recommended to inherit from `torch.nn.Module`.

### 1. `__init__`
Initializes your neural network. Arguments passed here come from the `static_params` in your config AND the hyperparameter suggestions from Optuna.

```python
def __init__(self, input_dim: int, hidden_dim: int):
    super().__init__()
    self.net = nn.Linear(input_dim, hidden_dim)
```

### 2. `forward`
Standard PyTorch forward pass.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)
```

### 3. `train_batch`
This is your "training step". It receives a raw batch of data and is responsible for calculating loss.

**Signature**:
```python
def train_batch(
    self, 
    batch: dict[str, Any], 
    optimizer: torch.optim.Optimizer, 
    writer: Any, 
    global_step: int
) -> tuple[float, dict[str, float]]:
```

**Responsibilities**:
1.  Move data to device (if not automatically handled).
2.  Zero gradients.
3.  Forward pass.
4.  Calculate loss.
5.  Backward pass (`loss.backward()`).
6.  Optimizer step (`optimizer.step()`).
7.  (Optional) Log to Tensorboard via `writer`.
8.  Return the loss value (float) and a dictionary of metrics.

### 4. `validate`
This handles evaluation on the validation set.

**Signature**:
```python
def validate(
    self, 
    data_loader: torch.utils.data.DataLoader, 
    **kwargs
) -> dict[str, float]:
```

**Responsibilities**:
1.  Iterate over the `data_loader`.
2.  Compute aggregate metrics (e.g., Mean Squared Error, Accuracy) on the validation set.
3.  Return a dictionary of metrics.
    *   One of these keys must match the `objective.metric` in your `config.yaml`.

## Example: PCA Reconstructor

Here is a skeletal example of a model that reconstructs data from PCA components.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCAReconstructor(nn.Module):
    def __init__(self, pca_dim: int, gene_dim: int):
        super().__init__()
        self.linear = nn.Linear(pca_dim, gene_dim)

    def forward(self, x):
        return self.linear(x)

    def train_batch(self, batch, optimizer, writer, global_step):
        self.train()
        x = batch['pca_scores']
        y = batch['counts']
        
        optimizer.zero_grad()
        pred = self(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        
        return loss.item(), {'mse': loss.item()}

    def validate(self, data_loader, **kwargs):
        self.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in data_loader:
                x = batch['pca_scores']
                y = batch['counts']
                pred = self(x)
                total_loss += F.mse_loss(pred, y).item()
                n += 1
        return {'val_loss': total_loss / n if n > 0 else 0.0}
```

!!! tip
    You can find a full featured example in `examples/pca_reconstructor.py`.
