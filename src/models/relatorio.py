"""
Modelo Pydantic para o relatório jornalístico gerado pela LLM.

Garante que o output do DSPy seja estruturado, validado e auditável —
mesmo contrato de qualidade dos outros JSONs do pipeline.

Output: relatorio.json
"""

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Normalizador de vocabulário: corrige anglicismos e anotações internas do DSPy.
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
    """Substitui anglicismos e anotações conhecidas por equivalentes em português."""
    for pattern, replacement in _PT_ANGLICISMS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Corretor de concordância de número: substantivo singular + adjetivo plural.
#
# Abordagem: regex morfológico puro (sem rede, sem dependências extras).
#
# Sufixos cobertos (transformações inequívocas em português):
#   -ais  → -al   (emocionais→emocional, especiais→especial, naturais→natural)
#   -áveis → -ável (formidáveis→formidável, memoráveis→memorável)
#   -íveis → -ível (possíveis→possível, incríveis→incrível)
#
# Heurística de número do substantivo: palavras terminadas em -as/-os/-es/-ais/-eis
# são tratadas como plurais (ex: "corridas") → adjetivo plural deixado intacto.
# Palavras funcionais (preposições, conjunções, artigos, verbos curtos) são
# ignoradas para evitar falsos positivos do tipo "e emocionais".
# ---------------------------------------------------------------------------

_FUNCTION_WORDS: frozenset[str] = frozenset({
    "a", "as", "o", "os", "um", "uma", "uns", "umas",
    "e", "ou", "mas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "com",
    "sem", "sob", "sobre", "entre", "que", "se", "ao", "à",
    "foi", "é", "era", "ser", "ter", "há", "pela", "pelo",
    # Advérbios/preposições que terminam em -ntes/-ais e seriam falsos positivos
    "antes", "depois",
})

# (padrão que detecta o sufixo plural no final, sufixo singular substituto)
_ADJ_PLURAL_SUFFIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ais$"),   "al"),    # emocionais→emocional, especiais→especial
    (re.compile(r"áveis$"), "ável"),  # formidáveis→formidável
    (re.compile(r"íveis$"), "ível"),  # possíveis→possível
    (re.compile(r"ntes$"),  "nte"),   # dominantes→dominante, importantes→importante
]

# Captura: (palavra-substantivo)(espaço)(palavra terminada em sufixo plural de adj)
_NOUN_PLURAL_ADJ = re.compile(
    r"(\b[A-Za-záàãâéêíóôõúçÁÀÃÂÉÊÍÓÔÕÚÇ]+\b)"
    r"\s+"
    r"(\b[A-Za-záàãâéêíóôõúçÁÀÃÂÉÊÍÓÔÕÚÇ]+(?:ntes|ais|áveis|íveis)\b)",
    re.UNICODE,
)

_PLURAL_NOUN_ENDINGS: tuple[str, ...] = ("as", "os", "es", "ais", "eis", "is", "ns")


def _fix_number_agreement(text: str) -> str:
    """
    Detecta e corrige discordância de número: substantivo singular + adjetivo plural.

    Algoritmo por correspondência regex:
    1. Localiza pares (palavra_anterior, palavra_plural_adj).
    2. Se uma conjunção coordenativa ("e", "ou") preceder o substantivo imediatamente
       → ignora (dois sujeitos coordenados tornam o plural correto).
    3. Se palavra_anterior for palavra funcional → não corrige.
    4. Se palavra_anterior terminar em sufixo de plural → substantivo plural → não corrige.
    5. Caso contrário (substantivo singular) → converte adjetivo ao singular usando
       as transformações inequívocas de sufixo.
    6. Restaura a capitalização original do adjetivo.
    """
    def _replace(m: re.Match) -> str:
        noun, adj = m.group(1), m.group(2)

        # Guarda os 5 caracteres antes do substantivo para detectar coordenação:
        # "Norris e Piastri emocionais" → preceding = "s e " → endswith " e" → skip
        preceding = text[max(0, m.start() - 5): m.start()].rstrip().lower()
        if preceding.endswith((" e", " ou")):
            return m.group(0)

        if noun.lower() in _FUNCTION_WORDS or adj.lower() in _FUNCTION_WORDS:
            return m.group(0)

        if any(noun.lower().endswith(s) for s in _PLURAL_NOUN_ENDINGS):
            return m.group(0)  # substantivo plural → adjetivo plural está correto

        for suffix_pat, singular_suffix in _ADJ_PLURAL_SUFFIXES:
            if suffix_pat.search(adj.lower()):
                corrected_lower = suffix_pat.sub(singular_suffix, adj.lower())
                # Preserva capitalização da primeira letra do original
                corrected = (
                    corrected_lower[0].upper() + corrected_lower[1:]
                    if adj[0].isupper()
                    else corrected_lower
                )
                return f"{noun} {corrected}"

        return m.group(0)  # nenhuma transformação aplicável

    return _NOUN_PLURAL_ADJ.sub(_replace, text)


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
        """
        Pipeline de normalização pós-geração aplicado a todos os campos:
          1. _normalize_pt         — vocabulário: anglicismos e anotações DSPy
          2. _fix_number_agreement — gramática: concordância de número (singular/plural)
        Ambas as funções são puramente locais (regex), sem chamadas de rede.
        """
        for field in ("titulo", "lead", "narrativa", "destaques", "conclusao"):
            val = _normalize_pt(getattr(self, field))
            val = _fix_number_agreement(val)
            setattr(self, field, val)
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
