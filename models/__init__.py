"""
Models package for flood/drought prediction.

This package contains neural network models for time series prediction,
including LSTM architectures for hydrological forecasting.
"""

from .LSTMModel import LSTMModel, create_model

__all__ = ['LSTMModel', 'create_model']
