"""
Módulo de pré-processamento de dados de telemetria F1 usando SciPy.

Este módulo transforma dados brutos do FastF1 em sinais limpos e sincronizados
prontos para processamento de ML.
"""

from .interpolation import synchronize_telemetry
from .signal_processing import clean_signal
from .feature_engineering import calculate_statistical_features

__all__ = [
    "synchronize_telemetry",
    "clean_signal",
    "calculate_statistical_features",
]
