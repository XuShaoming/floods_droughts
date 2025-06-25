"""
LSTM Model for Flood/Drought Prediction

This module contains the LSTM model architecture for many-to-many time series prediction.
The model supports multi-task learning and includes proper weight initialization,
batch normalization, and dropout for regularization.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM model for many-to-many time series prediction.
    
    Architecture:
    - Multiple LSTM layers with dropout
    - Batch normalization
    - Linear output layer for multi-task learning
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 batch_first: bool = True):
        """
        Initialize LSTM model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of hidden units in LSTM layers
        num_layers : int
            Number of LSTM layers
        output_size : int
            Number of output targets
        dropout : float
            Dropout probability
        batch_first : bool
            If True, input shape is (batch, seq, feature)
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=batch_first,
            bidirectional=False
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Reshape for batch normalization
        # (batch_size, seq_len, hidden_size) -> (batch_size * seq_len, hidden_size)
        lstm_out_reshaped = lstm_out.contiguous().view(-1, self.hidden_size)
        
        # Apply batch normalization
        normalized = self.batch_norm(lstm_out_reshaped)
        
        # Apply dropout
        dropped = self.dropout(normalized)
        
        # Linear layer
        output = self.fc(dropped)
        
        # Reshape back to (batch_size, seq_len, output_size)
        output = output.view(batch_size, seq_len, self.output_size)
        
        return output
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden states."""
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell


def create_model(input_size, config):
    """
    Factory function to create LSTM model from configuration.
    
    Parameters:
    -----------
    input_size : int
        Number of input features
    config : dict
        Model configuration dictionary
        
    Returns:
    --------
    LSTMModel
        Initialized LSTM model
    """
    return LSTMModel(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=len(config.get('target_cols', ['streamflow'])),
        dropout=config['dropout']
    )


if __name__ == "__main__":
    # Test the model
    model = LSTMModel(input_size=6, hidden_size=64, num_layers=2, output_size=1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size, seq_len, input_size = 32, 30, 6
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
