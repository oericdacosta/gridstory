"""
Preparação de dados para Machine Learning.

Funções para imputação, encoding de variáveis categóricas e escalonamento de features
para uso com algoritmos do Scikit-learn.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer


def impute_missing_values(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    strategy: str = "mean",
    use_knn: bool = False,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Imputa valores faltantes em colunas numéricas usando SimpleImputer ou KNNImputer.

    Os algoritmos de ML do Scikit-learn (K-Means, DBSCAN, Isolation Forest) requerem
    dados completos sem valores NaN. Esta função preenche lacunas garantindo que a
    matriz de entrada X esteja completa.

    Args:
        df: DataFrame de entrada
        numeric_columns: Lista de colunas numéricas para imputar.
                        Se None, detecta automaticamente todas as colunas numéricas.
        strategy: Estratégia de imputação para SimpleImputer: 'mean', 'median', 'most_frequent'
        use_knn: Se True, usa KNNImputer (mais sofisticado, considera vizinhos)
        n_neighbors: Número de vizinhos para KNNImputer (padrão: 5)

    Returns:
        DataFrame com valores faltantes imputados

    Rationale:
        - SimpleImputer: Rápido, bom para poucos valores faltantes
        - KNNImputer: Mais preciso, considera padrões nos dados, recomendado para telemetria

    Example:
        >>> laps_df = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')
        >>> laps_imputed = impute_missing_values(
        ...     laps_df,
        ...     numeric_columns=['LapTime_seconds', 'Sector1Time_seconds'],
        ...     strategy='median'
        ... )
    """
    df = df.copy()

    # Detectar colunas numéricas se não especificadas
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        # Sem colunas numéricas para imputar
        return df

    # Selecionar apenas colunas que existem no DataFrame
    existing_columns = [col for col in numeric_columns if col in df.columns]

    if not existing_columns:
        return df

    # Escolher imputador
    if use_knn:
        imputer = KNNImputer(n_neighbors=n_neighbors)
    else:
        imputer = SimpleImputer(strategy=strategy)

    # Imputar valores
    df[existing_columns] = imputer.fit_transform(df[existing_columns])

    return df


def encode_categorical_variables(
    df: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    drop_first: bool = False,
) -> pd.DataFrame:
    """
    Codifica variáveis categóricas usando OneHotEncoder.

    Variáveis como Composto do Pneu (Soft, Medium, Hard) são qualitativas.
    O OneHotEncoder transforma essas categorias em colunas binárias (ex: is_soft, is_hard),
    permitindo que algoritmos de clusterização entendam que tempos de volta feitos com
    pneus diferentes pertencem a contextos diferentes.

    Args:
        df: DataFrame de entrada
        categorical_columns: Lista de colunas categóricas para codificar.
                           Se None, usa colunas padrão: ['Compound', 'TrackStatus']
        drop_first: Se True, remove a primeira categoria de cada coluna para evitar
                   multicolinearidade (dummy trap)

    Returns:
        DataFrame com variáveis categóricas codificadas em colunas binárias

    Rationale:
        - Algoritmos baseados em distância (K-Means, DBSCAN) não entendem categorias
        - OneHotEncoding cria features binárias que representam "presença" de uma categoria
        - Exemplo: Compound='SOFT' vira is_soft=1, is_medium=0, is_hard=0

    Example:
        >>> laps_df = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')
        >>> laps_encoded = encode_categorical_variables(
        ...     laps_df,
        ...     categorical_columns=['Compound'],
        ...     drop_first=True
        ... )
        >>> # Agora tem colunas: Compound_MEDIUM, Compound_HARD, Compound_SOFT
    """
    df = df.copy()

    # Usar colunas padrão se não especificadas
    if categorical_columns is None:
        categorical_columns = ['Compound', 'TrackStatus']

    # Selecionar apenas colunas que existem no DataFrame
    existing_columns = [col for col in categorical_columns if col in df.columns]

    if not existing_columns:
        return df

    # Usar pd.get_dummies para simplicidade (OneHotEncoder do pandas)
    # É equivalente ao sklearn.preprocessing.OneHotEncoder mas mais simples
    df = pd.get_dummies(
        df,
        columns=existing_columns,
        drop_first=drop_first,
        dtype=int
    )

    return df


def scale_features(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    scaler_type: str = "standard",
    return_scaler: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, StandardScaler | RobustScaler]:
    """
    Escalona features numéricas usando StandardScaler ou RobustScaler.

    Esta é a etapa MAIS CRÍTICA para algoritmos baseados em distância (K-Means, DBSCAN).
    Se você misturar "Tempo de Volta" (ex: 90 segundos) com "Idade do Pneu" (ex: 5 voltas),
    a variável de maior magnitude dominará o cálculo se não houver escalonamento.

    Args:
        df: DataFrame de entrada
        numeric_columns: Lista de colunas numéricas para escalonar.
                        Se None, detecta automaticamente colunas numéricas.
        scaler_type: Tipo de scaler:
                    - 'standard': StandardScaler (remove média, divide por std)
                    - 'robust': RobustScaler (baseado em quartis, resistente a outliers)
        return_scaler: Se True, retorna também o objeto scaler (para aplicar em novos dados)

    Returns:
        DataFrame com features escalonadas
        OU
        Tupla (DataFrame escalonado, scaler) se return_scaler=True

    Rationale:
        - StandardScaler: Padrão, assume distribuição aproximadamente normal
        - RobustScaler: Melhor quando há muitos outliers (ex: rodadas, colisões)
        - Coloca todas as variáveis na mesma escala (média 0, variância 1)

    Example:
        >>> laps_df = pd.read_parquet('data/processed/races/2025/round_01/laps_processed.parquet')
        >>> # Escalonar apenas colunas relevantes para clustering
        >>> laps_scaled = scale_features(
        ...     laps_df,
        ...     numeric_columns=['LapTime_seconds', 'TyreLife', 'Sector1Time_seconds'],
        ...     scaler_type='robust'  # Resistente a outliers
        ... )
    """
    df = df.copy()

    # Detectar colunas numéricas se não especificadas
    if numeric_columns is None:
        # Excluir colunas booleanas e binárias (já estão em 0/1)
        numeric_columns = []
        for col in df.select_dtypes(include=[np.number]).columns:
            # Pular colunas binárias (apenas 0 e 1)
            unique_vals = df[col].dropna().unique()
            if not (len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})):
                numeric_columns.append(col)

    if not numeric_columns:
        if return_scaler:
            return df, None
        return df

    # Selecionar apenas colunas que existem no DataFrame
    existing_columns = [col for col in numeric_columns if col in df.columns]

    if not existing_columns:
        if return_scaler:
            return df, None
        return df

    # Escolher scaler
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:  # default: standard
        scaler = StandardScaler()

    # Escalonar valores
    df[existing_columns] = scaler.fit_transform(df[existing_columns])

    if return_scaler:
        return df, scaler

    return df
