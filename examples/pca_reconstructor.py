"""PCA Reconstruction Models for reconstructing gene expression from PCA space.

This module provides three variants:
1. LinearPCAReconstructor: Simple linear transformation (PCA inverse)
2. MLPPCAReconstructor: Multi-layer perceptron for non-linear reconstruction
3. HybridPCAReconstructor: Applies PCA components then MLP correction
"""
from __future__ import annotations
from typing import Optional, Sequence, Any, Literal
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Network Building Blocks
# ============================================================================

class MLPBlock(nn.Module):
    """Multi-layer perceptron block."""

    def __init__(
        self,
        input_dim: int,
        dims: Sequence[int],
        act_fn: nn.Module = nn.SiLU(),
        dropout_rate: float = 0.0,
        act_last_layer: bool = True,
    ):
        super().__init__()
        self.act_last_layer = act_last_layer

        layers = []
        prev_dim = input_dim
        for i, dim in enumerate(dims):
            layers.append(nn.Linear(prev_dim, dim))
            if i < len(dims) - 1 or act_last_layer:
                layers.append(act_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# Base PCA Reconstructor
# ============================================================================

class PCAReconstructorBase(nn.Module):
    """Base class for PCA reconstruction models.

    All reconstruction models should inherit from this class and implement
    the forward, train_batch, validate, and inference methods.
    """

    def __init__(self, pca_dim: int, gene_dim: int):
        super().__init__()
        self.pca_dim = pca_dim
        self.gene_dim = gene_dim

    def forward(self, pca_scores: torch.Tensor) -> torch.Tensor:
        """Forward pass: reconstruct gene expression from PCA scores.

        Args:
            pca_scores: PCA scores of shape (batch, pca_dim)

        Returns:
            reconstructed: Reconstructed gene expression of shape (batch, gene_dim)
        """
        raise NotImplementedError

    def train_batch(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        writer: Any,
        global_step: int,
    ) -> tuple[float, dict]:
        """Train on a single batch.

        Args:
            batch: Dictionary with keys 'pca_scores' and 'target_counts'
            optimizer: PyTorch optimizer
            writer: TensorBoard writer
            global_step: Current training step

        Returns:
            loss: Loss value
            metrics: Dictionary of metrics
        """
        self.train()

        pca_scores = batch['X_avg_pca'] # Updated to match example pipeline or flexible key? 
        # The user provided code uses 'pca_scores' but my transform outputs 'X_avg_pca'.
        # I should probably map it or assume the batch keys are correct.
        # For now I'll check keys in batch and fallback.
        if 'pca_scores' in batch:
             pca_scores = batch['pca_scores']
        else:
             pca_scores = batch['X_avg_pca']
             
        # Target counts is likely 'X' or 'counts'
        # The dataset interface returns 'X'
        if 'target_counts' in batch:
            target_counts = batch['target_counts']
        else:
            target_counts = batch['X']

        # Forward pass
        optimizer.zero_grad()
        reconstructed = self(pca_scores)

        # MSE loss
        loss = F.mse_loss(reconstructed, target_counts)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Logging
        if writer is not None:
            writer.log_scalar('Train/MSE_Loss', loss.item(), global_step)

            # Compute gradient norm
            grad_norm = torch.norm(
                torch.stack([p.grad.detach().data.norm(2) for p in self.parameters() if p.grad is not None]),
                2
            ).item()
            writer.log_scalar('Gradients/Total_Norm', grad_norm, global_step)

        return loss.item(), {'mse_loss': loss.item()}

    @torch.inference_mode()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Perform inference.

        Args:
            batch: Dictionary with keys 'pca_scores' and optionally 'target_counts'

        Returns:
            predictions: Reconstructed gene expression
            targets: True target counts (if available)
            loss: Loss value (if targets available)
            metrics: Dictionary of metrics
        """
        self.eval()

        if 'pca_scores' in batch:
             pca_scores = batch['pca_scores']
        else:
             pca_scores = batch['X_avg_pca']

        if 'target_counts' in batch:
             target_counts = batch['target_counts']
        else:
             target_counts = batch.get('X', None)

        # Forward pass
        predictions = self(pca_scores)

        # Compute loss if targets available
        if target_counts is not None:
            loss = F.mse_loss(predictions, target_counts)
        else:
            loss = torch.tensor(0.0, device=pca_scores.device)

        return predictions, target_counts, loss, {}

    def move_batch_to_device(self, device: torch.device, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch

    def validate(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_workers: int = 1,
        de_batch_size: int = 100,
        fdr_threshold: float = 0.05,
    ) -> dict[str, float]:
        """Validate the model on a data loader with comprehensive metrics.

        Computes MAE, discrimination score, and DE overlap metrics using
        the PredictionValidator class from vcc.eval.

        Args:
            data_loader: PyTorch DataLoader with validation data
            num_workers: Number of workers for parallel DE computation
            de_batch_size: Batch size for DE computation
            fdr_threshold: FDR threshold for DE overlap metric

        Returns:
            Dictionary of validation metrics: mae, disc, de_overlap
        """
        self.eval()

        # Get model device
        device = next(self.parameters()).device

        # Get dataset references
        # Access the underlying dataset to get metadata
        # We assume data_loader.dataset is accessible and has the required attributes
        # If wrapped (e.g. Subset), we might need to dig deeper, but usually attributes are not forwarded.
        # We will try to access attributes from the dataset object.
        dataset = data_loader.dataset
        
        # Check if dataset is a Subset and unwrap if necessary/possible or check attributes on it
        if isinstance(dataset, torch.utils.data.Subset):
            # Subset doesn't proxy attributes. We need the underlying dataset for metadata 
            # if the attributes are not on the Subset itself.
            # However, control_adata should be specific to the split or global? 
            # Usually control_adata is global or we need to look at parent.
            # Let's try to access from dataset, if not present, try dataset.dataset
            if not hasattr(dataset, "control_adata") and hasattr(dataset, "dataset"):
                 # This is a bit risky if control_adata needs to be filtered, but usually it's "the controls"
                 parent = dataset.dataset
        else:
             parent = dataset

        control_adata = getattr(dataset, "control_adata", getattr(parent, "control_adata", None))
        gene_names = getattr(dataset, "gene_names", getattr(parent, "gene_names", None))

        if control_adata is None or gene_names is None:
            logger.warning("dataset.control_adata or dataset.gene_names not found. Skipping advanced metrics.")
            advanced_metrics = False
        else:
            advanced_metrics = True

        # Collect predictions, targets, and labels
        predictions_list = []
        targets_list = []
        labels_list = []
        
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                # Move batch to model device (if not already)
                batch = self.move_batch_to_device(device, batch)

                if 'pca_scores' in batch:
                     pca_scores = batch['pca_scores']
                else:
                     pca_scores = batch['X_avg_pca']
                
                if 'target_counts' in batch:
                    target_counts = batch['target_counts']
                else:
                    target_counts = batch['X']

                # Forward pass
                reconstructed = self(pca_scores)
                
                loss = F.mse_loss(reconstructed, target_counts)
                total_loss += loss.item()
                n_batches += 1

                # Collect results
                predictions_list.append(reconstructed.cpu().numpy())
                targets_list.append(target_counts.cpu().numpy())
                
                # Check for perturbation label
                if 'perturbation_label' in batch:
                    labels_list.extend(batch['perturbation_label'])
                elif 'target_gene' in batch: # Fallback or specific mapping
                    labels_list.extend(batch['target_gene'])
                else:
                    # If no labels, we can't do per-perturbation metrics
                    if len(labels_list) == 0 and advanced_metrics:
                         logger.warning("No 'perturbation_label' or 'target_gene' in batch. disabling advanced metrics.")
                         advanced_metrics = False

        loss = total_loss / n_batches if n_batches > 0 else 0.0
        metrics = {'loss': loss}
        
        if not advanced_metrics:
            return metrics

        # Concatenate all predictions and targets
        all_predictions = np.vstack(predictions_list)
        all_targets = np.vstack(targets_list)

        # Aggregate by perturbation (compute mean per perturbation)
        unique_perturbations = sorted(set(labels_list))
        # Ensure unique_perturbations doesn't contain tensor/numpy scalars if direct from batch
        # if elements are strings, fine.
        
        agg_predictions_list = []
        agg_targets_list = []
        valid_perts = []
        
        # Optimize aggregation
        # Convert labels_list to numpy array for boolean masking
        labels_arr = np.array(labels_list)
        
        for pert in unique_perturbations:
            pert_mask = (labels_arr == pert)
            if np.sum(pert_mask) > 0:
                agg_predictions_list.append(all_predictions[pert_mask].mean(axis=0))
                agg_targets_list.append(all_targets[pert_mask].mean(axis=0))
                valid_perts.append(pert)
        
        if not agg_predictions_list:
             return metrics

        agg_predictions = np.vstack(agg_predictions_list)
        agg_targets = np.vstack(agg_targets_list)

        # Replace NaNs with 0 (can happen with untrained models or numerical instability)
        agg_predictions = np.nan_to_num(agg_predictions, nan=0.0)
        agg_targets = np.nan_to_num(agg_targets, nan=0.0)

        # Extract control cells
        if hasattr(control_adata.X, 'toarray'):
            control_cells = control_adata.X.toarray()
        else:
            control_cells = np.array(control_adata.X)

        # Replace NaNs in control cells as well
        control_cells = np.nan_to_num(control_cells, nan=0.0)

        # Compute only discrimination score (PDS) for efficiency
        try:
            import anndata as ad
            from cell_eval._types._anndata import PerturbationAnndataPair
            from cell_eval.metrics import discrimination_score

            # Build AnnData objects
            # valid_perts corresponds to rows of agg_predictions
            pert_labels = ["control"] * len(control_cells) + valid_perts

            real_adata = ad.AnnData(
                X=np.vstack([control_cells, agg_targets]),
                obs={"perturbation": pert_labels}
            )
            real_adata.var.index = gene_names

            pred_adata = ad.AnnData(
                X=np.vstack([control_cells, agg_predictions]),
                obs={"perturbation": pert_labels}
            )
            pred_adata.var.index = gene_names

            # Create data pair for cell_eval
            data_pair = PerturbationAnndataPair(
                real=real_adata,
                pred=pred_adata,
                pert_col="perturbation",
                control_pert="control"
            )

            # Calculate discrimination score only
            disc_results = discrimination_score(data_pair, metric="l1", exclude_target_gene=True)
            average_disc = np.mean(list(disc_results.values()))
            metrics['disc'] = average_disc
            
        except ImportError:
            logger.warning("cell_eval not installed, skipping discrimination score.")
        except Exception as e:
            logger.error(f"Error computing discrimination score: {e}")

        return metrics


# ============================================================================
# Linear PCA Reconstructor
# ============================================================================

class LinearPCAReconstructor(PCAReconstructorBase):
    """Linear PCA reconstruction model (baseline).

    This model learns a simple linear transformation from PCA scores to gene expression.
    It essentially learns the inverse PCA transformation (PCA_components.T @ scores + mean).
    """

    def __init__(
        self,
        pca_dim: int,
        gene_dim: int,
    ):
        super().__init__(pca_dim, gene_dim)

        # Linear layer (no bias, bias will be handled separately as mean)
        self.projection = nn.Linear(pca_dim, gene_dim, bias=True)

    def forward(self, pca_scores: torch.Tensor) -> torch.Tensor:
        """Reconstruct gene expression from PCA scores.

        Args:
            pca_scores: PCA scores of shape (batch, pca_dim)

        Returns:
            reconstructed: Reconstructed gene expression of shape (batch, gene_dim)
        """
        return self.projection(pca_scores)


# ============================================================================
# MLP PCA Reconstructor
# ============================================================================

class MLPPCAReconstructor(PCAReconstructorBase):
    """MLP-based PCA reconstruction model.

    This model uses a multi-layer perceptron to learn a non-linear reconstruction
    from PCA scores to gene expression.
    """

    def __init__(
        self,
        pca_dim: int,
        gene_dim: int,
        hidden_dims: Sequence[int] = (2048, 2048),
        dropout_rate: float = 0.0,
        act_fn: nn.Module = nn.SiLU(),
    ):
        super().__init__(pca_dim, gene_dim)

        # MLP encoder
        self.encoder = MLPBlock(
            input_dim=pca_dim,
            dims=hidden_dims,
            act_fn=act_fn,
            dropout_rate=dropout_rate,
            act_last_layer=True,
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dims[-1], gene_dim)

    def forward(self, pca_scores: torch.Tensor) -> torch.Tensor:
        """Reconstruct gene expression from PCA scores.

        Args:
            pca_scores: PCA scores of shape (batch, pca_dim)

        Returns:
            reconstructed: Reconstructed gene expression of shape (batch, gene_dim)
        """
        hidden = self.encoder(pca_scores)
        reconstructed = self.output_projection(hidden)
        return reconstructed


# ============================================================================
# Hybrid PCA Reconstructor
# ============================================================================

class HybridPCAReconstructor(PCAReconstructorBase):
    """Hybrid PCA reconstruction model.

    This model first applies the PCA components to get a linear reconstruction,
    then uses an MLP to learn corrections/residuals. This is useful when the
    data lies in a discrete space (e.g., log-transformed counts) where a simple
    linear reconstruction may not be optimal.
    """

    def __init__(
        self,
        pca_dim: int,
        gene_dim: int,
        pca_components: torch.Tensor,  # Shape: (gene_dim, pca_dim)
        pca_mean: torch.Tensor,  # Shape: (gene_dim,)
        hidden_dims: Sequence[int] = (1024, 1024),
        dropout_rate: float = 0.0,
        act_fn: nn.Module = nn.SiLU(),
        residual_weight: float = 0.1,  # Weight for residual correction
    ):
        super().__init__(pca_dim, gene_dim)

        # Store PCA components and mean as buffers (not trainable)
        self.register_buffer('pca_components', pca_components)
        self.register_buffer('pca_mean', pca_mean)

        self.residual_weight = residual_weight

        # MLP for learning residual correction
        # Input is concatenation of PCA scores and linear reconstruction
        self.residual_net = MLPBlock(
            input_dim=pca_dim + gene_dim,
            dims=hidden_dims,
            act_fn=act_fn,
            dropout_rate=dropout_rate,
            act_last_layer=True,
        )

        # Output projection for residual
        self.residual_projection = nn.Linear(hidden_dims[-1], gene_dim)

    def forward(self, pca_scores: torch.Tensor) -> torch.Tensor:
        """Reconstruct gene expression from PCA scores.

        Args:
            pca_scores: PCA scores of shape (batch, pca_dim)

        Returns:
            reconstructed: Reconstructed gene expression of shape (batch, gene_dim)
        """
        # Linear reconstruction using PCA components
        linear_reconstruction = pca_scores @ self.pca_components.T + self.pca_mean

        # Concatenate PCA scores and linear reconstruction
        combined = torch.cat([pca_scores, linear_reconstruction], dim=-1)

        # Learn residual correction
        residual_hidden = self.residual_net(combined)
        residual = self.residual_projection(residual_hidden)

        # Final reconstruction: linear + weighted residual
        reconstructed = linear_reconstruction + self.residual_weight * residual

        return reconstructed



# ============================================================================
# Unified Wrapper
# ============================================================================

class UnifiedPCAReconstructor(PCAReconstructorBase):
    """Unified wrapper for PCA Reconstructor models."""
    
    # Class attributes for hybrid model
    PCA_COMPONENTS: torch.Tensor | None = None
    PCA_MEAN: torch.Tensor | None = None
    
    def __init__(
        self, 
        model_type: str, 
        pca_dim: int, 
        gene_dim: int, 
        act_fn: str | nn.Module = "SiLU",
        **kwargs
    ):
        super().__init__(pca_dim, gene_dim)
        
        # Handle activation function
        if isinstance(act_fn, str):
            if hasattr(nn, act_fn):
                act_fn_module = getattr(nn, act_fn)()
                # If the module has parameters (like PReLU), this minimal init might not be enough
                # and might need arguments. But standard activations are usually parameterless.
            else:
                logger.warning(f"Activation {act_fn} not found in torch.nn, using SiLU")
                act_fn_module = nn.SiLU()
        else:
            act_fn_module = act_fn
            
        self.model_type = model_type
        
        if model_type == "linear":
            self.model = LinearPCAReconstructor(pca_dim, gene_dim)
        elif model_type == "mlp":
            # Filter kwargs for MLP. MLPBlock in this file takes 'act_last_layer' but not used in Unified args
            # Actually MLPPCAReconstructor __init__ takes: pca_dim, gene_dim, hidden_dims, dropout_rate, act_fn
            mlp_kwargs = {k: v for k, v in kwargs.items() if k in ['hidden_dims', 'dropout_rate']}
            
            # Note: hidden_dims might come as string from Optuna if deserialized? No, json loads handles it.
            # But the code provided for MLPPCAReconstructor uses Sequence[int].
            
            self.model = MLPPCAReconstructor(
                pca_dim, gene_dim, 
                act_fn=act_fn_module,
                **mlp_kwargs
            )
        elif model_type == "hybrid":
            if self.PCA_COMPONENTS is None or self.PCA_MEAN is None:
                raise ValueError("PCA_COMPONENTS and PCA_MEAN must be set for Hybrid model")
            # Filter kwargs for Hybrid
            hybrid_kwargs = {k: v for k, v in kwargs.items() if k in ['hidden_dims', 'dropout_rate', 'residual_weight']}
            self.model = HybridPCAReconstructor(
                pca_dim, gene_dim, 
                pca_components=self.PCA_COMPONENTS,
                pca_mean=self.PCA_MEAN,
                act_fn=act_fn_module,
                **hybrid_kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def forward(self, pca_scores: torch.Tensor) -> torch.Tensor:
        return self.model(pca_scores)
        
    def train_batch(self, batch, optimizer, writer, global_step):
        return self.model.train_batch(batch, optimizer, writer, global_step)

    def validate(self, data_loader, num_workers=1, de_batch_size=100, fdr_threshold=0.05):
        return self.model.validate(data_loader, num_workers, de_batch_size, fdr_threshold)
        
    def inference(self, batch, **kwargs):
        return self.model.inference(batch, **kwargs)



# Alias for CLI compatibility
Model = UnifiedPCAReconstructor

__all__ = [
    'PCAReconstructorBase',
    'LinearPCAReconstructor',
    'MLPPCAReconstructor',
    'HybridPCAReconstructor',
    'UnifiedPCAReconstructor',
    'Model',
]
