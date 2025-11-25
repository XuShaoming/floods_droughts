"""
Hierarchical LSTM for global multi-watershed training with static attributes.

Extends HierarchicalLSTMModel by adding an optional static attribute encoder
mirroring the CTLSTM global model. Intermediate predictions feed back into the
final LSTM inputs, allowing gradients from the final targets to propagate
through intermediate branches.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class CTLSTMGlobal(nn.Module):
    """
    Hierarchical LSTM with intermediate branches plus optional static embeddings.
    """

    def __init__(
        self,
        input_size: int,
        intermediate_targets: Optional[List[str]],
        final_targets: List[str],
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        batch_first: bool = True,
        static_input_size: int = 0,
        static_embedding_layers: Optional[List[int]] = None,
        static_dropout: float = 0.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.intermediate_targets = intermediate_targets or []
        self.final_targets = final_targets
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.batch_first = batch_first

        self.n_intermediate = len(self.intermediate_targets)
        self.n_final = len(self.final_targets)
        self.static_input_size = static_input_size

        # Static encoder (mirrors CTLSTM implementation)
        self.static_encoder: Optional[nn.Sequential] = None
        self.static_embedding_dim = 0
        if self.static_input_size > 0:
            layers: List[nn.Module] = []
            in_features = self.static_input_size
            target_layers = static_embedding_layers or [self.static_input_size]
            for size in target_layers:
                layers.append(nn.Linear(in_features, size))
                layers.append(nn.ReLU())
                if static_dropout > 0:
                    layers.append(nn.Dropout(static_dropout))
                in_features = size
            self.static_encoder = nn.Sequential(*layers)
            self.static_embedding_dim = in_features

        combined_dynamic_size = input_size + self.static_embedding_dim

        self.intermediate_lstms = nn.ModuleDict()
        self.intermediate_batch_norms = nn.ModuleDict()
        self.intermediate_dropouts = nn.ModuleDict()
        self.intermediate_outputs = nn.ModuleDict()

        for target_name in self.intermediate_targets:
            self.intermediate_lstms[target_name] = nn.LSTM(
                input_size=combined_dynamic_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=batch_first,
            )
            self.intermediate_batch_norms[target_name] = nn.BatchNorm1d(hidden_size)
            self.intermediate_dropouts[target_name] = nn.Dropout(dropout)
            self.intermediate_outputs[target_name] = nn.Linear(hidden_size, 1)

        final_input_size = combined_dynamic_size + self.n_intermediate
        self.final_lstm = nn.LSTM(
            input_size=final_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=batch_first,
        )
        self.final_batch_norm = nn.BatchNorm1d(hidden_size)
        self.final_dropout = nn.Dropout(dropout)
        self.final_output = nn.Linear(hidden_size, self.n_final)

        self._init_weights()

    def _init_weights(self):
        for target_name in self.intermediate_targets:
            lstm = self.intermediate_lstms[target_name]
            for name, param in lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    param.data[(n // 4) : (n // 2)].fill_(1)
            nn.init.xavier_uniform_(self.intermediate_outputs[target_name].weight)
            nn.init.zeros_(self.intermediate_outputs[target_name].bias)

        for name, param in self.final_lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n // 4) : (n // 2)].fill_(1)

        nn.init.xavier_uniform_(self.final_output.weight)
        nn.init.zeros_(self.final_output.bias)

    def _concat_static(self, x: torch.Tensor, static_inputs: Optional[torch.Tensor]) -> torch.Tensor:
        if self.static_encoder is None:
            return x
        if static_inputs is None or static_inputs.numel() == 0:
            raise ValueError("Static inputs must be provided when static_input_size > 0.")
        if static_inputs.dim() != 2 or static_inputs.size(1) != self.static_input_size:
            raise ValueError(
                f"Expected static input shape (batch, {self.static_input_size}), got {tuple(static_inputs.shape)}"
            )
        static_emb = self.static_encoder(static_inputs)
        batch_size, seq_len, _ = x.size()
        expanded = static_emb.unsqueeze(1).expand(-1, seq_len, -1)
        return torch.cat([x, expanded], dim=-1)

    def forward(self, x: torch.Tensor, static_inputs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        dynamic_with_static = self._concat_static(x, static_inputs)

        intermediate_predictions: Dict[str, torch.Tensor] = {}
        intermediate_stack: List[torch.Tensor] = []

        for target_name in self.intermediate_targets:
            lstm_out, _ = self.intermediate_lstms[target_name](dynamic_with_static)
            reshaped = lstm_out.contiguous().view(-1, self.hidden_size)
            normalized = self.intermediate_batch_norms[target_name](reshaped)
            dropped = self.intermediate_dropouts[target_name](normalized)
            output = self.intermediate_outputs[target_name](dropped)
            output = output.view(batch_size, seq_len, 1)
            intermediate_predictions[target_name] = output
            intermediate_stack.append(output)

        if intermediate_stack:
            intermediate_concat = torch.cat(intermediate_stack, dim=-1)
        else:
            intermediate_concat = torch.zeros(batch_size, seq_len, 0, device=x.device, dtype=x.dtype)

        final_input = torch.cat([dynamic_with_static, intermediate_concat], dim=-1)
        final_out, _ = self.final_lstm(final_input)
        reshaped_final = final_out.contiguous().view(-1, self.hidden_size)
        normalized_final = self.final_batch_norm(reshaped_final)
        dropped_final = self.final_dropout(normalized_final)
        final_predictions = self.final_output(dropped_final).view(batch_size, seq_len, self.n_final)

        return {"intermediate": intermediate_predictions, "final": final_predictions}
