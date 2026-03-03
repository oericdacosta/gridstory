"""
DSPy Examples e Métricas de Avaliação para Otimização com MIPROv2.

Parte 3 do planejamento: Dataset de exemplos + métricas + MIPROv2.

Pré-requisito: rodar pipeline para R02–R05 e curar relatórios gold.

Uso:
    # Criar exemplos curados
    example = make_example(
        race_summary=race_summary,
        timeline=timeline,
        driver_profiles=driver_profiles,
        gold_relatorio=gold_relatorio_dict,
    )

    # Avaliar um predict
    score = metric_relatorio(example, pred)

    # Otimizar com MIPROv2
    from src.llm.examples import optimize_reporter
    optimize_reporter(trainset=examples, output_path="models/reporter_optimized.json")
"""

import re
from pathlib import Path

import dspy

from src.llm.narrative_context import build_narrative_context
from src.models.relatorio import RelatorioSecoes


# ---------------------------------------------------------------------------
# Formato de Exemplo DSPy
# ---------------------------------------------------------------------------

def make_example(
    race_summary: dict,
    timeline: list,
    driver_profiles: list,
    gold_relatorio: dict,
    race_control: list | None = None,
    driver_quality_scores: dict | None = None,
) -> dspy.Example:
    """
    Cria um exemplo DSPy a partir de dados da corrida e de um relatório gold curado.

    O relatório gold deve ser um dict com os campos de RelatorioSecoes:
    titulo, lead, desenvolvimento, analise_tecnica, destaques_individuais, conclusao.

    Args:
        race_summary:          Conteúdo de race_summary.json
        timeline:              Conteúdo de timeline.json (list de eventos)
        driver_profiles:       Conteúdo de driver_profiles.json
        gold_relatorio:        Dict com o relatório jornalístico curado manualmente
        race_control:          Mensagens do race_control (opcional)
        driver_quality_scores: Scores de qualidade por piloto (opcional)

    Returns:
        dspy.Example com campo 'contexto' (input) e 'relatorio' (gold output)
    """
    contexto = build_narrative_context(
        race_summary=race_summary,
        timeline=timeline,
        driver_profiles=driver_profiles,
        race_control=race_control,
        driver_quality_scores=driver_quality_scores,
    )

    relatorio = RelatorioSecoes(**gold_relatorio)

    return dspy.Example(
        contexto=contexto,
        relatorio=relatorio,
    ).with_inputs("contexto")


# ---------------------------------------------------------------------------
# Métricas de Avaliação (usadas pelo MIPROv2)
# ---------------------------------------------------------------------------

def _word_count(relatorio: RelatorioSecoes) -> int:
    """Total de palavras em todos os campos do relatório."""
    return relatorio.word_count() if hasattr(relatorio, "word_count") else sum(
        len(f.split())
        for f in [
            relatorio.titulo,
            relatorio.lead,
            relatorio.desenvolvimento,
            relatorio.analise_tecnica,
            relatorio.destaques_individuais,
            relatorio.conclusao,
        ]
    )


def _coverage_score(contexto: str, relatorio: RelatorioSecoes) -> float:
    """
    Mede cobertura dos eventos do contexto no relatório.

    Verifica se safety cars, undercuts, overcuts e abandonos mencionados
    no contexto aparecem no desenvolvimento e análise técnica.
    """
    try:
        import json
        ctx = json.loads(contexto)
        events = ctx.get("events", [])
    except Exception:
        return 1.0

    relevant_types = {"safety_car", "undercut", "overcut", "abandono", "queda_de_pneu"}
    relevant = [e for e in events if e.get("type") in relevant_types]
    if not relevant:
        return 1.0

    text = (relatorio.desenvolvimento + " " + relatorio.analise_tecnica).lower()
    covered = sum(
        1 for evt in relevant
        if str(evt.get("lap", "")) in text
        or f"volta {evt.get('lap', '')}" in text
    )
    return covered / len(relevant)


def _no_technical_terms_score(relatorio: RelatorioSecoes) -> float:
    """
    Penaliza uso de termos técnicos de ML que não devem aparecer no artigo.

    Termos proibidos: tire cliff, drop_seconds, push %, anomaly, etc.
    """
    forbidden = re.compile(
        r"\b(tire cliff|drop_seconds|push %|base %|degraded %|anomaly|"
        r"cliff_validated|ataque %|desgaste %|cluster|isolation forest|"
        r"SOFT|MEDIUM|HARD|INTERMEDIATE)\b",
        re.IGNORECASE,
    )
    full_text = " ".join([
        relatorio.titulo,
        relatorio.lead,
        relatorio.desenvolvimento,
        relatorio.analise_tecnica,
        relatorio.destaques_individuais,
        relatorio.conclusao,
    ])
    matches = len(forbidden.findall(full_text))
    return max(0.0, 1.0 - matches * 0.1)


def _winner_mentioned_score(contexto: str, relatorio: RelatorioSecoes) -> float:
    """Verifica se o vencedor da corrida é mencionado no lead."""
    try:
        import json
        ctx = json.loads(contexto)
        winner = ctx.get("race", {}).get("winner", "")
        # winner pode ser "Nome Completo (COD)"
        winner_code = re.search(r"\(([A-Z]{3})\)", winner)
        if winner_code:
            code = winner_code.group(1)
            if code in relatorio.lead or code in relatorio.titulo:
                return 1.0
        if winner and winner.split("(")[0].strip().split()[-1].lower() in relatorio.lead.lower():
            return 1.0
    except Exception:
        pass
    return 0.0


def metric_relatorio(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    """
    Métrica composta para avaliação do relatório gerado pelo DSPy.

    Composição:
        - 30%: Vencedor mencionado no lead/título
        - 40%: Cobertura de eventos da timeline no relatório
        - 20%: Ausência de termos técnicos de ML
        - 10%: Comprimento adequado (500–800 palavras)

    Args:
        example: DSPy Example com 'contexto' e 'relatorio' gold
        pred:    DSPy Prediction com campo 'relatorio'
        trace:   Contexto de trace do DSPy (não usado)

    Returns:
        Score entre 0.0 (péssimo) e 1.0 (perfeito)
    """
    try:
        relatorio: RelatorioSecoes = pred.relatorio
    except AttributeError:
        return 0.0

    score = 0.0

    # 30%: Vencedor mencionado
    score += _winner_mentioned_score(example.contexto, relatorio) * 0.30

    # 40%: Cobertura de eventos
    score += _coverage_score(example.contexto, relatorio) * 0.40

    # 20%: Sem termos técnicos
    score += _no_technical_terms_score(relatorio) * 0.20

    # 10%: Comprimento adequado
    wc = _word_count(relatorio)
    if 500 <= wc <= 800:
        score += 0.10
    elif 400 <= wc < 500 or 800 < wc <= 1000:
        score += 0.05

    return round(score, 3)


# ---------------------------------------------------------------------------
# Otimização com MIPROv2
# ---------------------------------------------------------------------------

def optimize_reporter(
    trainset: list[dspy.Example],
    output_path: str | Path = "models/reporter_optimized.json",
    auto: str = "medium",
) -> None:
    """
    Otimiza o GeradorRelatorio com MIPROv2 usando os exemplos curados.

    Pré-requisitos:
        1. trainset com pelo menos 10 exemplos (20-30 ideal)
        2. DSPy configurado com o LM (chamar setup_lm() antes)

    Args:
        trainset:    Lista de dspy.Example com 'contexto' e 'relatorio'
        output_path: Caminho para salvar o programa otimizado (.json)
        auto:        Modo automático do MIPROv2: 'light', 'medium' ou 'heavy'

    Example:
        >>> from src.llm.reporter import setup_lm
        >>> from src.llm.examples import optimize_reporter, make_example
        >>> setup_lm()
        >>> examples = [make_example(...) for _ in races]
        >>> optimize_reporter(trainset=examples)
    """
    from dspy.teleprompt import MIPROv2
    from src.llm.reporter import GeradorRelatorio

    if len(trainset) < 5:
        raise ValueError(
            f"trainset com apenas {len(trainset)} exemplos — mínimo 5 para MIPROv2 "
            "(recomendado: 10-30). Rode o pipeline para R02-R05 e cure os relatórios."
        )

    optimizer = MIPROv2(
        metric=metric_relatorio,
        auto=auto,
        verbose=True,
    )

    optimized = optimizer.compile(
        GeradorRelatorio(),
        trainset=trainset,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(output_path))
    print(f"[MIPROv2] Programa otimizado salvo: {output_path}")
