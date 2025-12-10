# ruff: noqa: PGH004
# ruff: noqa
# mypy: ignore-errors
"""This file contains the PyTorch model for the performance test of the Titanic dataset."""

from typing import Any

import torch


class ModelTitanicPerformance(torch.nn.Module):
    def __init__(self, in_features: int = 16, out_features: int = 1, lyrs: list = [8]):
        super(ModelTitanicPerformance, self).__init__()
        activation_linear = torch.nn.Identity()
        activation_output = torch.nn.Identity()
        layers = []
        for i in range(len(lyrs)):
            layers.append(torch.nn.Linear(in_features, lyrs[i]))
            layers.append(activation_linear)
            in_features = lyrs[i]
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(activation_output)
        self.layers = torch.nn.ModuleList(layers)

        # Define loss function internally (new protocol)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(
        self,
        Age: torch.Tensor,
        Family_Size: torch.Tensor,
        Fare: torch.Tensor,
        Parch: torch.Tensor,
        Pclass: torch.Tensor,
        Sex: torch.Tensor,
        SibSp: torch.Tensor,
        Embarked_C: torch.Tensor,
        Embarked_Q: torch.Tensor,
        Embarked_S: torch.Tensor,
        Title_Dr: torch.Tensor,
        Title_Master: torch.Tensor,
        Title_Miss: torch.Tensor,
        Title_Mr: torch.Tensor,
        Title_Mrs: torch.Tensor,
        Title_Rev: torch.Tensor,
        **kwargs: torch.Tensor,  # noqa: ARG002
    ):
        x = torch.stack(
            [
                Age,
                Family_Size,
                Fare,
                Parch,
                Pclass,
                Sex,
                SibSp,
                Embarked_C,
                Embarked_Q,
                Embarked_S,
                Title_Dr,
                Title_Master,
                Title_Miss,
                Title_Mr,
                Title_Mrs,
                Title_Rev,
            ],
            dim=1,
        )
        x = x.squeeze(-1)
        for layer in self.layers:
            x = layer(x)
        return x

    def compute_loss(self, output: torch.Tensor, Survived: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            output: Model output tensor of shape [batch_size, nb_classes]
            survived: Target tensor of shape [batch_size, 1]

        Returns:
            Loss value
        """
        # Squeeze the extra dimension from the target tensor and ensure long dtype
        target = Survived.squeeze(-1)

        # Ensure both tensors have the same shape and dimensionality
        if output.dim() == 0:
            output = output.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)

        return self.loss_fn(output, target)

    def compute_accuracy(self, output: torch.Tensor, Survived: torch.Tensor) -> torch.Tensor:
        """Compute the accuracy.

        Args:
            output: Model output tensor of shape [batch_size, nb_classes]
            survived: Target tensor of shape [batch_size, 1]
        """
        # Squeeze the extra dimension from the target tensor and ensure long dtype
        target = Survived.squeeze(-1)
        # Compute the accuracy
        accuracy = ((output > 0) == target).float().mean()
        return accuracy

    def train_batch(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        logger: Any,  # ExperimentLogger
        global_step: int,
    ) -> tuple[float, dict[str, float]]:
        """Perform one training batch step.

        Args:
            batch: Dictionary with the input and label tensors
            optimizer: Optimizer for the training step
            logger: ExperimentLogger for metrics logging
            global_step: Current training step

        Returns:
            Tuple of (loss value, metrics dictionary)
        """
        # Forward pass
        output = self.forward(**batch).squeeze(-1)

        # Compute loss
        loss = self.compute_loss(output, batch["Survived"])

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        accuracy = self.compute_accuracy(output, batch["Survived"])

        # Log metrics
        logger.log_scalar("train/loss", loss.item(), global_step)
        logger.log_scalar("train/accuracy", accuracy.item(), global_step)

        return loss.item(), {"accuracy": accuracy.item()}

    def validate(
        self,
        data_loader: torch.utils.data.DataLoader,
        logger: Any | None = None,
        global_step: int | None = None,
    ) -> dict[str, float]:
        """Validate the model on a data loader.

        Args:
            data_loader: Validation data loader (batches already on correct device)
            logger: Optional logger for metrics logging
            global_step: Optional step for logging

        Returns:
            Dictionary of validation metrics
        """
        self.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            device = next(self.parameters()).device
            for batch in data_loader:
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                output = self.forward(**batch).squeeze(-1)

                # Compute loss and metrics
                loss = self.compute_loss(output, batch["Survived"])
                accuracy = self.compute_accuracy(output, batch["Survived"])

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1

        # Average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0

        # Log if logger provided
        if logger and global_step is not None:
            logger.log_scalar("val/loss", avg_loss, global_step)
            logger.log_scalar("val/accuracy", avg_accuracy, global_step)

        return {"loss": avg_loss, "accuracy": avg_accuracy}
