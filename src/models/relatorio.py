"""
Modelo Pydantic para o relatório jornalístico gerado pela LLM.

Garante que o output do DSPy seja estruturado, validado e auditável —
mesmo contrato de qualidade dos outros JSONs do pipeline.

Output: relatorio.json
"""

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Normalizador pós-geração: corrige anglicismos que o modelo comete ao
# escrever em português sobre F1 (vocabulário predominantemente inglês).
# Aplicado via model_validator após a criação do objeto, não no prompt.
# ---------------------------------------------------------------------------

_PT_ANGLICISMS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bdeployado\b", re.IGNORECASE), "acionado"),
    (re.compile(r"\bdeployada\b", re.IGNORECASE), "acionada"),
    (re.compile(r"\bdeployed\b", re.IGNORECASE), "acionado"),
    (re.compile(r"\bdeployar\b", re.IGNORECASE), "acionar"),
    (re.compile(r"\bdeployment\b", re.IGNORECASE), "acionamento"),
    (re.compile(r"\bdeploy\b", re.IGNORECASE), "acionamento"),
]


def _normalize_pt(text: str) -> str:
    """Replaces known F1 anglicisms with proper Portuguese equivalents."""
    for pattern, replacement in _PT_ANGLICISMS:
        text = pattern.sub(replacement, text)
    return text


class RelatorioSecoes(BaseModel):
    """
    Relatório jornalístico de uma corrida de F1 dividido em seções canônicas.

    Gerado pelo DSPy (Predict) e validado pelo Pydantic antes de
    ser serializado. Cada campo corresponde a uma seção do artigo.

    Estrutura intencional:
      - titulo + lead: manchete e abertura
      - narrativa: corpo único do artigo (substituiu os campos separados
        desenvolvimento e analise_tecnica para evitar estrutura artificial)
      - destaques: pilotos individualmente
      - conclusao: encerramento
    """

    model_config = ConfigDict(extra="forbid")

    titulo: str = Field(
        ...,
        min_length=10,
        description="Título impactante e factual do artigo.",
    )
    lead: str = Field(
        ...,
        min_length=50,
        description="Contexto e resultado da corrida em 2-3 frases diretas.",
    )
    narrativa: str = Field(
        ...,
        min_length=150,
        description="Corpo principal: narrativa cronológica integrando acontecimentos e estratégia.",
    )
    destaques: str = Field(
        ...,
        min_length=50,
        description="Pilotos com performances notáveis, positivas ou negativas.",
    )
    conclusao: str = Field(
        ...,
        min_length=20,
        description="Frase de encerramento sobre o resultado da corrida.",
    )

    @model_validator(mode="after")
    def normalize_portuguese(self) -> "RelatorioSecoes":
        """Corrects known anglicisms in all text fields after generation."""
        self.titulo = _normalize_pt(self.titulo)
        self.lead = _normalize_pt(self.lead)
        self.narrativa = _normalize_pt(self.narrativa)
        self.destaques = _normalize_pt(self.destaques)
        self.conclusao = _normalize_pt(self.conclusao)
        return self

    def word_count(self) -> int:
        """Total de palavras em todas as seções do artigo."""
        all_text = " ".join([
            self.titulo,
            self.lead,
            self.narrativa,
            self.destaques,
            self.conclusao,
        ])
        return len(all_text.split())

    def to_text(self) -> str:
        """Texto corrido sem formatação (para logging/MLflow)."""
        return (
            f"{self.titulo}\n\n"
            f"{self.lead}\n\n"
            f"{self.narrativa}\n\n"
            f"{self.destaques}\n\n"
            f"{self.conclusao}"
        )

    def to_markdown(self) -> str:
        """Artigo formatado em Markdown para exibição final."""
        return (
            f"# {self.titulo}\n\n"
            f"> {self.lead}\n\n"
            f"{self.narrativa}\n\n"
            f"---\n\n"
            f"{self.destaques}\n\n"
            f"{self.conclusao}"
        )
