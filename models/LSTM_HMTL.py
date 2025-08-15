"""
Hierarchical LSTM Model for Flood/Drought Prediction (HMTL/HSTL)

This module contains the hierarchical LSTM model architecture for many-to-many time series prediction
with intermediate targets. The model supports both Hierarchical Sequential Task Learning (HSTL) and
Hierarchical Multi-Task Learning (HMTL).

Architecture:
- Multiple intermediate LSTM branches (one per intermediate target)
- Final LSTM layer that combines original features with intermediate predictions
- Support for freezing/unfreezing layers for sequential training
- Dynamic architecture based on number of intermediate targets
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple


class HierarchicalLSTMModel(nn.Module):
    """
    Hierarchical LSTM model for many-to-many time series prediction with intermediate targets.
    
    Architecture:
    - Intermediate LSTM branches: Each predicts one intermediate target from original features
    - Final LSTM layer: Predicts final targets using original features + intermediate predictions
    - Support for both HSTL (sequential) and HMTL (joint) training strategies
    """
    
    def __init__(self, 
                 input_size: int,
                 intermediate_targets: List[str],
                 final_targets: List[str] = ["streamflow"],
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 batch_first: bool = True):
        """
        Initialize Hierarchical LSTM model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features (original meteorological variables)
        intermediate_targets : List[str]
            List of intermediate target names (e.g., ["PET", "ET", "SUPY", ...])
        final_targets : List[str]
            List of final target names (e.g., ["streamflow"])
        hidden_size : int
            Number of hidden units in LSTM layers
        num_layers : int
            Number of LSTM layers for each branch
        dropout : float
            Dropout probability
        batch_first : bool
            If True, input shape is (batch, seq, feature)
        """
        super(HierarchicalLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.intermediate_targets = intermediate_targets
        self.final_targets = final_targets
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.n_intermediate = len(intermediate_targets)
        self.n_final = len(final_targets)
        
        # Create intermediate LSTM branches (one per intermediate target)
        self.intermediate_lstms = nn.ModuleDict()
        self.intermediate_batch_norms = nn.ModuleDict()
        self.intermediate_dropouts = nn.ModuleDict()
        self.intermediate_outputs = nn.ModuleDict()
        
        for target_name in intermediate_targets:
            # LSTM layer for this intermediate target
            self.intermediate_lstms[target_name] = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=batch_first,
                bidirectional=False
            )
            
            # Batch normalization for this intermediate target
            self.intermediate_batch_norms[target_name] = nn.BatchNorm1d(hidden_size)
            
            # Dropout for this intermediate target
            self.intermediate_dropouts[target_name] = nn.Dropout(dropout)
            
            # Output layer for this intermediate target (single output)
            self.intermediate_outputs[target_name] = nn.Linear(hidden_size, 1)
        
        # Final LSTM layer (takes original features + intermediate predictions)
        final_input_size = input_size + self.n_intermediate
        self.final_lstm = nn.LSTM(
            input_size=final_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=batch_first,
            bidirectional=False
        )
        
        # Final layers
        self.final_batch_norm = nn.BatchNorm1d(hidden_size)
        self.final_dropout = nn.Dropout(dropout)
        self.final_output = nn.Linear(hidden_size, self.n_final)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize intermediate LSTM weights
        for target_name in self.intermediate_targets:
            lstm = self.intermediate_lstms[target_name]
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    param.data[(n//4):(n//2)].fill_(1)
            
            # Initialize intermediate output layer
            nn.init.xavier_uniform_(self.intermediate_outputs[target_name].weight)
            nn.init.zeros_(self.intermediate_outputs[target_name].bias)
        
        # Initialize final LSTM weights
        for name, param in self.final_lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        # Initialize final output layer
        nn.init.xavier_uniform_(self.final_output.weight)
        nn.init.zeros_(self.final_output.bias)
    
    def forward(self, x, training_mode='joint'):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
        training_mode : str
            Training mode: 'joint', 'intermediate_only', 'final_only'
            
        Returns:
        --------
        dict
            Dictionary containing predictions:
            - 'intermediate': Dict[str, torch.Tensor] - Intermediate predictions
            - 'final': torch.Tensor - Final predictions
        """
        batch_size, seq_len, _ = x.size()
        
        # Forward pass through intermediate LSTM branches
        intermediate_predictions = {}
        intermediate_outputs_list = []
        
        for target_name in self.intermediate_targets:
            # LSTM forward pass
            lstm_out, _ = self.intermediate_lstms[target_name](x)
            
            # Reshape for batch normalization
            lstm_out_reshaped = lstm_out.contiguous().view(-1, self.hidden_size)
            
            # Apply batch normalization and dropout
            normalized = self.intermediate_batch_norms[target_name](lstm_out_reshaped)
            dropped = self.intermediate_dropouts[target_name](normalized)
            
            # Output layer
            output = self.intermediate_outputs[target_name](dropped)
            
            # Reshape back to (batch_size, seq_len, 1)
            output = output.view(batch_size, seq_len, 1)
            
            intermediate_predictions[target_name] = output
            intermediate_outputs_list.append(output)
        
        # Concatenate intermediate predictions for final LSTM input
        # Shape: (batch_size, seq_len, n_intermediate)
        intermediate_concat = torch.cat(intermediate_outputs_list, dim=-1)
        
        # Combine original features with intermediate predictions
        # Shape: (batch_size, seq_len, input_size + n_intermediate)
        final_input = torch.cat([x, intermediate_concat], dim=-1)
        
        # Forward pass through final LSTM
        final_lstm_out, _ = self.final_lstm(final_input)
        
        # Reshape for batch normalization
        final_lstm_out_reshaped = final_lstm_out.contiguous().view(-1, self.hidden_size)
        
        # Apply batch normalization and dropout
        final_normalized = self.final_batch_norm(final_lstm_out_reshaped)
        final_dropped = self.final_dropout(final_normalized)
        
        # Final output layer
        final_output = self.final_output(final_dropped)
        
        # Reshape back to (batch_size, seq_len, n_final)
        final_predictions = final_output.view(batch_size, seq_len, self.n_final)
        
        # Convert final predictions to dictionary format for consistency
        final_predictions_dict = {}
        for i, target_name in enumerate(self.final_targets):
            final_predictions_dict[target_name] = final_predictions[:, :, i:i+1]
        
        return {
            'intermediate': intermediate_predictions,
            'final': final_predictions_dict
        }
    
    def freeze_intermediate_layers(self):
        """Freeze all intermediate LSTM layers and outputs."""
        for target_name in self.intermediate_targets:
            # Freeze LSTM parameters
            for param in self.intermediate_lstms[target_name].parameters():
                param.requires_grad = False
            
            # Freeze batch norm parameters
            for param in self.intermediate_batch_norms[target_name].parameters():
                param.requires_grad = False
            
            # Freeze output layer parameters
            for param in self.intermediate_outputs[target_name].parameters():
                param.requires_grad = False
        
        print("Intermediate layers frozen")
    
    def unfreeze_intermediate_layers(self):
        """Unfreeze all intermediate LSTM layers and outputs."""
        for target_name in self.intermediate_targets:
            # Unfreeze LSTM parameters
            for param in self.intermediate_lstms[target_name].parameters():
                param.requires_grad = True
            
            # Unfreeze batch norm parameters
            for param in self.intermediate_batch_norms[target_name].parameters():
                param.requires_grad = True
            
            # Unfreeze output layer parameters
            for param in self.intermediate_outputs[target_name].parameters():
                param.requires_grad = True
        
        print("Intermediate layers unfrozen")
    
    def freeze_final_layer(self):
        """Freeze final LSTM layer and output."""
        # Freeze final LSTM parameters
        for param in self.final_lstm.parameters():
            param.requires_grad = False
        
        # Freeze final batch norm parameters
        for param in self.final_batch_norm.parameters():
            param.requires_grad = False
        
        # Freeze final output layer parameters
        for param in self.final_output.parameters():
            param.requires_grad = False
        
        print("Final layer frozen")
    
    def unfreeze_final_layer(self):
        """Unfreeze final LSTM layer and output."""
        # Unfreeze final LSTM parameters
        for param in self.final_lstm.parameters():
            param.requires_grad = True
        
        # Unfreeze final batch norm parameters
        for param in self.final_batch_norm.parameters():
            param.requires_grad = True
        
        # Unfreeze final output layer parameters
        for param in self.final_output.parameters():
            param.requires_grad = True
        
        print("Final layer unfrozen")
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        intermediate_params = 0
        for target_name in self.intermediate_targets:
            intermediate_params += sum(p.numel() for p in self.intermediate_lstms[target_name].parameters() if p.requires_grad)
            intermediate_params += sum(p.numel() for p in self.intermediate_batch_norms[target_name].parameters() if p.requires_grad)
            intermediate_params += sum(p.numel() for p in self.intermediate_outputs[target_name].parameters() if p.requires_grad)
        
        final_params = sum(p.numel() for p in self.final_lstm.parameters() if p.requires_grad)
        final_params += sum(p.numel() for p in self.final_batch_norm.parameters() if p.requires_grad)
        final_params += sum(p.numel() for p in self.final_output.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'intermediate': intermediate_params,
            'final': final_params
        }


def create_hierarchical_model(input_size, config):
    """
    Factory function to create Hierarchical LSTM model from configuration.
    
    Parameters:
    -----------
    input_size : int
        Number of input features
    config : dict
        Model configuration dictionary
        
    Returns:
    --------
    HierarchicalLSTMModel
        Initialized Hierarchical LSTM model
    """
    return HierarchicalLSTMModel(
        input_size=input_size,
        intermediate_targets=config.get('intermediate_targets', []),
        final_targets=config.get('target_cols', ['streamflow']),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

if __name__ == "__main__":
    # Test the hierarchical model
    print("Testing Hierarchical LSTM Model:")
    
    intermediate_targets = ["PET", "ET", "SUPY", "WYIE", "SNOW", "TWS", "LZS", "AGW"]
    model = HierarchicalLSTMModel(
        input_size=6, 
        intermediate_targets=intermediate_targets,
        final_targets=["streamflow"],
        hidden_size=64, 
        num_layers=2, 
        dropout=0.2
    )
    
    param_info = model.get_trainable_parameters()
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")
    print(f"Intermediate parameters: {param_info['intermediate']:,}")
    print(f"Final parameters: {param_info['final']:,}")
    
    # Test forward pass
    batch_size, seq_len, input_size = 32, 30, 6
    x = torch.randn(batch_size, seq_len, input_size)
    
    print(f"\nInput shape: {x.shape}")
    
    # Joint training mode
    output = model(x, training_mode='joint')
    print(f"Final output shape: {output['final'].shape}")
    print(f"Number of intermediate targets: {len(output['intermediate'])}")
    for target_name, pred in output['intermediate'].items():
        print(f"  {target_name} shape: {pred.shape}")
    
    # Test freezing/unfreezing
    print(f"\nTesting freeze/unfreeze functionality:")
    model.freeze_final_layer()
    param_info = model.get_trainable_parameters()
    print(f"After freezing final layer - Trainable: {param_info['trainable']:,}")
    
    model.unfreeze_final_layer()
    model.freeze_intermediate_layers()
    param_info = model.get_trainable_parameters()
    print(f"After freezing intermediate layers - Trainable: {param_info['trainable']:,}")
    
    model.unfreeze_intermediate_layers()
    param_info = model.get_trainable_parameters()
    print(f"After unfreezing all - Trainable: {param_info['trainable']:,}")
    
    print("\n" + "="*50)
    print("Testing Original LSTM Model:")
    
    # Test the original model
    original_model = LSTMModel(input_size=6, hidden_size=64, num_layers=2, output_size=1)
    print(f"Original model parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    
    # Test forward pass
    original_output = original_model(x)
    print(f"Original model input shape: {x.shape}")
    print(f"Original model output shape: {original_output.shape}")
