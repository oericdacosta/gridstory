"""
Módulo de configuração centralizada.

Carrega configurações do arquivo config.yaml na raiz do projeto.
"""

import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Classe para carregar e acessar configurações do projeto."""

    _instance = None
    _config = None

    def __new__(cls):
        """Singleton para garantir uma única instância de configuração."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Inicializa e carrega as configurações."""
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Carrega o arquivo config.yaml."""
        # Caminho para config.yaml (3 níveis acima: utils -> src -> raiz)
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Obtém um valor de configuração usando notação de pontos.

        Args:
            key_path: Caminho da chave usando pontos (ex: 'data.raw.races_dir')
            default: Valor padrão se a chave não existir

        Returns:
            Valor da configuração ou default

        Exemplo:
            >>> config = Config()
            >>> config.get('data.raw.races_dir')
            'data/raw/races'
        """
        keys = key_path.split(".")
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    @property
    def data_dirs(self) -> Dict[str, str]:
        """Retorna todos os diretórios de dados."""
        return self._config.get("data", {})

    @property
    def cache_dir(self) -> str:
        """Retorna o diretório de cache do FastF1."""
        cache_dir = self.get("cache.dir", "~/.cache/fastf1")
        # Expandir ~ para home do usuário
        return str(Path(cache_dir).expanduser())

    @property
    def extraction_config(self) -> Dict[str, Any]:
        """Retorna configurações de extração."""
        return self._config.get("extraction", {})

    @property
    def polling_config(self) -> Dict[str, int]:
        """Retorna configurações de polling."""
        return self.get("extraction.polling", {})

    @property
    def ml_config(self) -> Dict[str, Any]:
        """Retorna configurações de ML."""
        return self._config.get("ml", {})

    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        """Retorna configurações de pré-processamento."""
        return self._config.get("preprocessing", {})

    @property
    def api_config(self) -> Dict[str, Any]:
        """Retorna configurações da API."""
        return self._config.get("api", {})

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Retorna configurações de LLM."""
        return self._config.get("llm", {})

    # Getters específicos para pré-processamento
    def get_num_points(self) -> int:
        """Retorna número de pontos para interpolação."""
        return self.get("preprocessing.interpolation.num_points", 5000)

    def get_median_filter_kernel_size(self) -> int:
        """Retorna tamanho do kernel para filtro mediano."""
        return self.get("preprocessing.signal_processing.median_filter_kernel_size", 5)

    def get_savgol_kernel_size(self) -> int:
        """Retorna tamanho do kernel para filtro Savitzky-Golay."""
        return self.get("preprocessing.signal_processing.savgol_kernel_size", 11)

    def get_savgol_polyorder(self) -> int:
        """Retorna ordem polinomial para filtro Savitzky-Golay."""
        return self.get("preprocessing.signal_processing.savgol_polyorder", 3)

    def get_outlier_threshold(self) -> float:
        """Retorna threshold para detecção de outliers."""
        return self.get("preprocessing.signal_processing.outlier_threshold", 3.0)

    def get_imputation_strategy(self) -> str:
        """Retorna estratégia de imputação."""
        return self.get("preprocessing.imputation.strategy", "median")

    def get_use_knn_imputation(self) -> bool:
        """Retorna se deve usar KNN para imputação."""
        return self.get("preprocessing.imputation.use_knn", False)

    def get_encoding_drop_first(self) -> bool:
        """Retorna se deve dropar primeira categoria no encoding."""
        return self.get("preprocessing.encoding.drop_first", True)

    def get_scaling_type(self) -> str:
        """Retorna tipo de escalonamento."""
        return self.get("preprocessing.scaling.type", "robust")

    # Getters específicos para ML
    def get_random_state(self) -> int:
        """Retorna random state global para reprodutibilidade."""
        return self.get("ml.random_state", 42)

    def get_contamination(self, profile: str = "normal") -> float:
        """
        Retorna valor de contaminação para anomaly detection.

        Args:
            profile: Perfil de corrida - "clean", "normal", ou "chaotic"

        Returns:
            Valor de contaminação (padrão: 0.05)
        """
        # Primeiro tentar pegar do perfil específico
        contamination = self.get(
            f"ml.anomaly.contamination_profiles.{profile}", None
        )
        if contamination is not None:
            return contamination

        # Fallback para contamination padrão
        return self.get("ml.anomaly.contamination", 0.05)

    def get_n_estimators(self) -> int:
        """Retorna número de estimadores para Isolation Forest."""
        return self.get("ml.anomaly.n_estimators", 100)

    def get_k_range_min(self) -> int:
        """Retorna número mínimo de clusters."""
        return self.get("ml.clustering.k_range_min", 2)

    def get_k_range_max(self) -> int:
        """Retorna número máximo de clusters."""
        return self.get("ml.clustering.k_range_max", 6)

    def get_dbscan_min_samples(self) -> int:
        """Retorna min_samples para DBSCAN."""
        return self.get("ml.dbscan.min_samples", 3)

    def get_dbscan_eps(self) -> float:
        """Retorna eps para DBSCAN."""
        return self.get("ml.dbscan.eps", 0.5)

    # Getters específicos para Ruptures (change point detection)
    def get_ruptures_algorithm(self) -> str:
        """Retorna algoritmo para detecção de change points."""
        return self.get("ml.degradation.algorithm", "Pelt")

    def get_ruptures_model(self) -> str:
        """Retorna modelo de custo para Ruptures."""
        return self.get("ml.degradation.model", "l2")

    def get_ruptures_penalty(self) -> float:
        """Retorna penalidade para o algoritmo PELT."""
        return self.get("ml.degradation.penalty", 3)

    def get_ruptures_min_size(self) -> int:
        """Retorna número mínimo de voltas entre dois breakpoints."""
        return self.get("ml.degradation.min_size", 3)

    def get_ruptures_jump(self) -> int:
        """Retorna step do grid de busca do PELT."""
        return self.get("ml.degradation.jump", 1)

    def get_ruptures_min_cliff_magnitude(self) -> float:
        """Retorna magnitude mínima positiva (segundos) para considerar um cliff válido.
        Cliffs com magnitude < threshold são falsos positivos (ex: race start transition)."""
        return self.get("ml.degradation.min_cliff_magnitude", 0.3)

    def get_ruptures_penalty_search_range(self) -> list:
        """Retorna range de penalties para calibração via --penalty-search."""
        return self.get("ml.degradation.penalty_search_range", [1, 2, 3, 5, 8, 13, 21])

    def get_ruptures_validation_enabled(self) -> bool:
        """Retorna se a validação por slope está habilitada."""
        return self.get("ml.degradation.validation.enabled", True)

    def get_ruptures_validation_window(self) -> int:
        """Retorna janela de laps antes do cliff para calcular slope de degradação."""
        return self.get("ml.degradation.validation.window_laps", 5)

    def get_ruptures_validation_slope_threshold(self) -> float:
        """Retorna slope mínimo positivo (segundos/volta) para validar um cliff.
        Slope positivo = ritmo degradando = cliff real."""
        return self.get("ml.degradation.validation.slope_threshold", 0.05)

    def get_silhouette_threshold(self) -> float:
        """Retorna threshold de Silhouette para avaliação de qualidade de clustering.
        Para dados contínuos de F1, 0.25 é realista (range esperado: 0.2–0.4)."""
        return self.get("ml.clustering.evaluation.silhouette_threshold", 0.25)

    def get_davies_bouldin_threshold(self) -> float:
        """Retorna threshold máximo de Davies-Bouldin para clustering de qualidade."""
        return self.get("ml.clustering.evaluation.davies_bouldin_threshold", 1.5)

    # Getters específicos para MLFlow
    def get_mlflow_enabled(self) -> bool:
        """Retorna se o tracking MLFlow está habilitado."""
        return self.get("mlflow.enabled", False)

    def get_mlflow_tracking_uri(self) -> str:
        """Retorna URI do servidor de tracking MLFlow.

        Caminhos relativos "file:./" são resolvidos em absolutos a partir da
        raiz do projeto (diretório do config.yaml), garantindo que o MLflow
        escreva sempre no mesmo lugar independente do CWD do processo chamador.
        """
        uri = self.get("mlflow.tracking_uri", "file:./mlruns")
        if uri.startswith("file:./") or uri.startswith("file:../"):
            relative_part = uri[len("file:"):]
            project_root = Path(__file__).parent.parent.parent
            uri = "file:" + str((project_root / relative_part).resolve())
        return uri

    def get_mlflow_experiment_prefix(self) -> str:
        """Retorna prefixo do nome do experimento MLFlow."""
        return self.get("mlflow.experiment_prefix", "F1")

    def __getitem__(self, key: str) -> Any:
        """Permite acesso via colchetes."""
        return self.get(key)

    def __repr__(self) -> str:
        """Representação em string da configuração."""
        return f"Config(loaded_keys={list(self._config.keys())})"


# Instância global de configuração
config = Config()


def get_config() -> Config:
    """
    Função auxiliar para obter a instância de configuração.

    Returns:
        Instância singleton de Config

    Exemplo:
        >>> from src.utils.config import get_config
        >>> cfg = get_config()
        >>> print(cfg.get('data.raw.races_dir'))
        data/raw/races
    """
    return config
