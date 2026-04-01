"""FedRGBD — FedBN Strategy for Flower.

FedBN keeps BatchNorm layers local (not aggregated) while averaging all other layers.
Reference: Li et al., "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization" (ICLR 2021)

Usage: Import and use in server.py alongside existing FedAvg/FedProx strategies.
"""

import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _is_batchnorm_key(key: str) -> bool:
    """Check if a parameter key belongs to a BatchNorm layer.
    
    MobileNetV3 BatchNorm layers have keys containing:
    - '.weight' and '.bias' for BN gamma/beta
    - '.running_mean' and '.running_var' for BN statistics
    - '.num_batches_tracked' for tracking
    
    We identify BN layers by checking for these BN-specific state dict patterns.
    In MobileNetV3-Small, BN layers appear as e.g.:
      features.0.1.weight, features.0.1.bias, features.0.1.running_mean, etc.
    """
    bn_indicators = [
        'running_mean', 'running_var', 'num_batches_tracked',
    ]
    # Direct BN statistics — always local
    for indicator in bn_indicators:
        if indicator in key:
            return True
    return False


def _is_batchnorm_param(key: str) -> bool:
    """Check if key is a BN learnable parameter (gamma/beta).
    
    We need the model's layer names to distinguish BN weight/bias from Conv/Linear weight/bias.
    For MobileNetV3-Small, BN layers are at positions like:
      features.X.Y.weight/bias where Y is the BN index in a Sequential block.
    
    This function uses a heuristic: if the key ends with .weight or .bias AND
    the tensor is 1-dimensional, it's a BN parameter (since Conv weights are 4D 
    and Linear weights are 2D).
    
    However, since we receive numpy arrays without shape context here, we handle
    this in the aggregation function where we have access to shapes.
    """
    return _is_batchnorm_key(key)


class FedBN(FedAvg):
    """Federated Learning with Local Batch Normalization (FedBN).
    
    Inherits from FedAvg but excludes BatchNorm parameters from aggregation.
    Each client keeps its own BN statistics and learnable BN parameters.
    """
    
    def __init__(
        self,
        model_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize FedBN strategy.
        
        Args:
            model_keys: Ordered list of model state_dict keys. If None, will be
                        inferred from the first client's parameters.
            **kwargs: Arguments passed to FedAvg (min_fit_clients, etc.)
        """
        super().__init__(**kwargs)
        self.model_keys = model_keys
        self._bn_indices = None  # Will be computed once we know the keys
    
    def _compute_bn_indices(self, keys: List[str], param_shapes: Optional[List[tuple]] = None):
        """Determine which parameter indices correspond to BatchNorm layers."""
        bn_indices = set()
        for i, key in enumerate(keys):
            # Always exclude running_mean, running_var, num_batches_tracked
            if _is_batchnorm_key(key):
                bn_indices.add(i)
                continue
            # For .weight and .bias: check if 1D (BN) vs multi-D (Conv/Linear)
            # BN weight/bias keys in MobileNetV3 follow pattern: features.X.1.weight
            # where the '1' position is the BN layer in the ConvBNActivation block
            if param_shapes and i < len(param_shapes):
                if (key.endswith('.weight') or key.endswith('.bias')) and len(param_shapes[i]) == 1:
                    bn_indices.add(i)
        
        return bn_indices
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results, excluding BatchNorm parameters."""
        
        if not results:
            return None, {}
        
        # Extract parameters from all clients
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Infer model keys from parameter shapes if not provided
        if self._bn_indices is None:
            first_params = weights_results[0][0]
            param_shapes = [p.shape for p in first_params]
            
            if self.model_keys is None:
                # Generate synthetic keys based on MobileNetV3-Small structure
                # We identify BN by dimensionality: 1D = BN, else = non-BN
                self.model_keys = [f"param_{i}" for i in range(len(first_params))]
                self._bn_indices = set()
                for i, shape in enumerate(param_shapes):
                    if len(shape) == 1:
                        # Could be BN weight/bias OR classifier bias
                        # Classifier bias is the very last 1D param
                        # For safety, check if it's small (BN channels) vs 2 (num_classes)
                        # Actually, we need a better heuristic.
                        # Let's just use shape: BN layers match channel dims
                        self._bn_indices.add(i)
                # Remove the last 1D param (classifier bias) from BN set
                # In MobileNetV3-Small, the last parameter is classifier[-1].bias (shape [2])
                last_1d_indices = [i for i in self._bn_indices]
                if last_1d_indices:
                    # The classifier bias (shape [num_classes]) should not be excluded
                    # Find it: it's typically the last 1D tensor
                    for idx in sorted(last_1d_indices, reverse=True):
                        if param_shapes[idx] == (2,):  # num_classes = 2
                            self._bn_indices.discard(idx)
                            break
            else:
                self._bn_indices = self._compute_bn_indices(self.model_keys, param_shapes)
            
            n_total = len(first_params)
            n_bn = len(self._bn_indices)
            print(f"  [FedBN] Parameters: {n_total} total, {n_bn} BN (kept local), "
                  f"{n_total - n_bn} aggregated")
        
        # Perform weighted averaging ONLY for non-BN parameters
        # For BN parameters, keep the first client's values (they won't be used anyway
        # since each client keeps its own BN params)
        num_examples_total = sum(num_examples for _, num_examples in weights_results)
        
        # Start with zeros
        aggregated = [np.zeros_like(weights_results[0][0][i]) 
                      for i in range(len(weights_results[0][0]))]
        
        for client_params, num_examples in weights_results:
            weight = num_examples / num_examples_total
            for i, param in enumerate(client_params):
                if i not in self._bn_indices:
                    # Aggregate non-BN parameters (weighted average)
                    aggregated[i] += param * weight
                # BN parameters: leave as zeros (each client uses its own)
        
        # For BN parameters, use the first client's values as placeholder
        # (clients will ignore these and keep their local BN params)
        first_client_params = weights_results[0][0]
        for i in self._bn_indices:
            aggregated[i] = first_client_params[i].copy()
        
        # Aggregate metrics
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(fit_metrics)
        
        return ndarrays_to_parameters(aggregated), metrics_aggregated
