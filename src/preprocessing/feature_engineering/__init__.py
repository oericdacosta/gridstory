"""
Módulo de engenharia de features para análise de F1.

Este módulo foi modularizado em 3 submódulos organizados por responsabilidade:
- statistical: Features estatísticas e análise de degradação
- domain: Pré-processamento específico de F1 (race control, weather, results)
- ml_prep: Preparação de dados para ML (imputação, encoding, scaling)

Todas as funções são re-exportadas aqui para manter backward compatibility com
código existente que importa diretamente do módulo principal.
"""

# Importar tudo dos submódulos
from .statistical import (
    calculate_statistical_features,
    calculate_degradation_rate,
    calculate_descriptive_statistics,
    enrich_dataframe_with_stats,
)

from .domain import (
    preprocess_race_control,
    preprocess_weather,
    preprocess_results,
)

from .ml_prep import (
    impute_missing_values,
    encode_categorical_variables,
    scale_features,
)

# Lista de exports públicos
__all__ = [
    # Funções estatísticas
    "calculate_statistical_features",
    "calculate_degradation_rate",
    "calculate_descriptive_statistics",
    "enrich_dataframe_with_stats",
    # Pré-processamento de domínio F1
    "preprocess_race_control",
    "preprocess_weather",
    "preprocess_results",
    # Preparação para ML
    "impute_missing_values",
    "encode_categorical_variables",
    "scale_features",
]
