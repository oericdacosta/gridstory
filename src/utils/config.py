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

        with open(config_path, 'r', encoding='utf-8') as f:
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
        keys = key_path.split('.')
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
        return self._config.get('data', {})

    @property
    def cache_dir(self) -> str:
        """Retorna o diretório de cache do FastF1."""
        cache_dir = self.get('cache.dir', '~/.cache/fastf1')
        # Expandir ~ para home do usuário
        return str(Path(cache_dir).expanduser())

    @property
    def extraction_config(self) -> Dict[str, Any]:
        """Retorna configurações de extração."""
        return self._config.get('extraction', {})

    @property
    def polling_config(self) -> Dict[str, int]:
        """Retorna configurações de polling."""
        return self.get('extraction.polling', {})

    @property
    def ml_config(self) -> Dict[str, Any]:
        """Retorna configurações de ML."""
        return self._config.get('ml', {})

    @property
    def api_config(self) -> Dict[str, Any]:
        """Retorna configurações da API."""
        return self._config.get('api', {})

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Retorna configurações de LLM."""
        return self._config.get('llm', {})

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
