"""FedRGBD — FedBN Strategy for Flower (v2, name-based BN detection).

FedBN keeps BatchNorm layers local (not aggregated) while averaging all other layers.
Reference: Li et al., "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization" (ICLR 2021)

This version identifies BN layers by name using the actual model state_dict keys,
which is more reliable than shape-based heuristics. Also fixes int64 dtype issue
with num_batches_tracked buffers.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import sys
sys.path.insert(0, ".")
from src.models.mobilenetv3_multimodal import create_model


def get_bn_indices_from_model():
    """Get the indices of BatchNorm parameters in MobileNetV3-Small state_dict.

    Returns a set of integer indices corresponding to BN layer parameters.
    Uses the actual model to determine which layers are BatchNorm instances.
    """
    import torch.nn as nn

    model = create_model(num_classes=2, in_channels=3, pretrained=False)

    # Find names of all BN layers in the model
    bn_layer_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layer_names.add(name)

    # Now find indices in state_dict that belong to these BN layers
    bn_indices = set()
    state_dict_keys = list(model.state_dict().keys())

    for i, key in enumerate(state_dict_keys):
        # key format: "features.0.1.weight" — strip the final suffix
        parent = key.rsplit('.', 1)[0]
        if parent in bn_layer_names:
            bn_indices.add(i)

    return bn_indices, len(state_dict_keys)


class FedBN(FedAvg):
    """Federated Learning with Local Batch Normalization (FedBN).

    Excludes BatchNorm parameters from aggregation. BN indices are determined
    once at initialization using the actual model architecture.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bn_indices, self._total_params = get_bn_indices_from_model()
        print(f"  [FedBN init] Parameters: {self._total_params} total, "
              f"{len(self._bn_indices)} BN (kept local), "
              f"{self._total_params - len(self._bn_indices)} aggregated")

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

        first_client_params = weights_results[0][0]
        num_params = len(first_client_params)
        num_examples_total = sum(num_examples for _, num_examples in weights_results)

        # Sanity check
        if num_params != self._total_params:
            print(f"  [FedBN WARNING] Expected {self._total_params} params, got {num_params}")

        # Initialize aggregated list
        aggregated = []
        for i in range(num_params):
            if i in self._bn_indices:
                # BN parameters: keep first client's values as placeholder
                # (clients will ignore these and use their own local BN)
                aggregated.append(first_client_params[i].copy())
            else:
                # Non-BN: float zeros for weighted averaging
                # Cast integer params to float32 to avoid dtype cast errors
                p = first_client_params[i]
                if np.issubdtype(p.dtype, np.integer):
                    aggregated.append(np.zeros(p.shape, dtype=np.float32))
                else:
                    aggregated.append(np.zeros_like(p))

        # Weighted averaging for non-BN parameters
        for client_params, num_examples in weights_results:
            weight = num_examples / num_examples_total
            for i in range(num_params):
                if i not in self._bn_indices:
                    aggregated[i] = aggregated[i] + client_params[i].astype(aggregated[i].dtype) * weight

        # Aggregate metrics (use fit_metrics_aggregation_fn if set)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(aggregated), metrics_aggregated
