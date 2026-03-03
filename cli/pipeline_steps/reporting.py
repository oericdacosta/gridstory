"""
Módulo de reporting para formatação de saídas do pipeline.

Fornece classes e funções para impressão formatada consistente durante
a execução do pipeline completo.
"""

import pandas as pd


class Reporter:
    """Classe para formatação consistente de mensagens do pipeline."""

    def __init__(self, phase_name: str, verbose: bool = False):
        """
        Inicializa o reporter.

        Args:
            phase_name: Nome da fase (ex: "EXTRAÇÃO", "PRÉ-PROCESSAMENTO", "MACHINE LEARNING")
            verbose: Se deve mostrar mensagens detalhadas
        """
        self.phase_name = phase_name
        self.verbose = verbose

    def header(self, title: str | None = None):
        """Imprime cabeçalho da fase."""
        print("\n" + "=" * 80)
        if title:
            print(title)
        else:
            print(f"📊 {self.phase_name}")
        print("=" * 80)

    def section(self, section_number: str, description: str):
        """
        Imprime cabeçalho de seção.

        Args:
            section_number: Número da seção (ex: "2.1", "3.2")
            description: Descrição da seção
        """
        print(f"\n🔄 {section_number} {description}...")

    def step(self, number: str, description: str):
        """
        Imprime etapa do processamento (com indentação).

        Args:
            number: Número da etapa
            description: Descrição da etapa
        """
        print(f"   {number}. {description}")

    def info(self, message: str, indent: int = 1):
        """
        Imprime mensagem informativa.

        Args:
            message: Mensagem a ser impressa
            indent: Nível de indentação (número de espaços triplos)
        """
        prefix = "   " * indent
        print(f"{prefix}📊 {message}")

    def success(self, message: str, indent: int = 1):
        """
        Imprime mensagem de sucesso.

        Args:
            message: Mensagem a ser impressa
            indent: Nível de indentação
        """
        prefix = "   " * indent
        print(f"{prefix}✅ {message}")

    def metric(self, label: str, value, indent: int = 2):
        """
        Imprime métrica (label: valor).

        Args:
            label: Nome da métrica
            value: Valor da métrica
            indent: Nível de indentação
        """
        prefix = "   " * indent
        print(f"{prefix}• {label}: {value}")

    def sample(self, df: pd.DataFrame, columns: list[str] | None = None, n: int = 5):
        """
        Imprime amostra do DataFrame.

        Args:
            df: DataFrame a ser amostrado
            columns: Colunas a mostrar (None = todas)
            n: Número de linhas a mostrar
        """
        if not self.verbose:
            return

        print("\n   📋 Amostra dos dados:")
        if columns:
            display_df = df[columns].head(n)
        else:
            display_df = df.head(n)
        print(display_df.to_string(index=False))

    def divider(self):
        """Imprime linha divisória."""
        print("-" * 80)


def print_pipeline_header(year: int, round_num: int):
    """
    Imprime cabeçalho principal do pipeline.

    Args:
        year: Ano da temporada
        round_num: Número da rodada
    """
    print("\n" + "=" * 80)
    print("🏎️  PITWALL AI - PIPELINE COMPLETO")
    print("=" * 80)
    print(f"📅 Temporada: {year}, Rodada: {round_num}")
    print("=" * 80)


def print_final_summary(race_dir, processed_dir, ml_dir, timeline_dir=None):
    """
    Imprime resumo final do pipeline.

    Args:
        race_dir: Diretório dos dados brutos
        processed_dir: Diretório dos dados processados
        ml_dir: Diretório dos resultados de ML
        timeline_dir: Diretório dos JSONs e relatório (Fases 4 e 5)
    """
    print("\n" + "=" * 80)
    print("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    print(f"📂 Dados brutos:      {race_dir}")
    print(f"📂 Dados processados: {processed_dir}")
    print(f"📂 Resultados ML:     {ml_dir}")
    if timeline_dir is not None:
        from pathlib import Path
        timeline_dir = Path(timeline_dir)
        print(f"📂 JSONs + Relatório: {timeline_dir}")
        relatorio = timeline_dir / "relatorio.json"
        if relatorio.exists():
            print(f"   └─ relatorio.json: {relatorio.stat().st_size // 1024}KB")
    print("=" * 80)
