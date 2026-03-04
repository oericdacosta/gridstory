"""
DSPy reporter module — generates structured journalistic F1 race reports.

Uses Groq (llama-3.3-70b-versatile) via DSPy Predict.
Input is pre-processed by NarrativeContext (~60% token reduction).
Output: 6 campos de texto independentes (titulo, lead, desenvolvimento,
analise_tecnica, destaques_individuais, conclusao) — montados em RelatorioSecoes
após geração. Campos separados evitam compressão do modelo num único JSON.
Full MLflow tracing: tokens, latency, tokens/s, hallucination_score, event_coverage.
"""

import os
import re
import time

import dspy
import mlflow
from dotenv import load_dotenv

from src.llm.narrative_context import build_narrative_context, _DRIVER_NAMES
from src.models.relatorio import RelatorioSecoes

load_dotenv()


# ---------------------------------------------------------------------------
# LM configuration
# ---------------------------------------------------------------------------

def setup_lm() -> dspy.LM:
    """Build and return a DSPy LM for Groq. Does NOT call dspy.configure() to
    avoid the thread-ownership restriction; callers must use dspy.context(lm=lm)."""
    from src.utils.config import config as _cfg

    provider = _cfg.get("llm.provider", "groq")
    model_name = _cfg.get("llm.model", "llama-3.3-70b-versatile")
    max_tokens = _cfg.get("llm.max_tokens", 1500)
    temperature = _cfg.get("llm.temperature", 0.3)

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Copy .env.example to .env and fill in your key."
            )
        lm = dspy.LM(
            model=f"groq/{model_name}",
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Provider não suportado: {provider}")

    return lm


# ---------------------------------------------------------------------------
# DSPy Signature — 6 campos de output independentes (evita compressão em JSON único)
# ---------------------------------------------------------------------------

class RelatorioJornalistico(dspy.Signature):
    """Você é um jornalista de F1. Escreva uma reportagem curta e envolvente em português do \
Brasil, baseada EXCLUSIVAMENTE nos dados em `contexto`. Tom: direto, humano e acessível — \
como se contasse a corrida a um amigo que não assistiu. Não invente dados.

REGRAS (invioláveis):
- Use nomes completos dos pilotos e equipes exatamente como no JSON.
- Pneus: "pneus macios/médios/duros/intermediários/de chuva". Nunca SOFT/MEDIUM/HARD/WET.
- Sem termos técnicos internos: tire cliff, drop_seconds, push %, anomaly, ataque %.
- Cubra os itens de must_cover.desenvolvimento e must_cover.analise_tecnica.
- Safety cars: cite volta e duração. Abandonos: cite volta e causa se disponível.
- ANTI-REPETIÇÃO: cada piloto e cada evento deve aparecer UMA ÚNICA VEZ no artigo inteiro \
(exceto vencedor no lead e conclusão). Nunca repita o mesmo abandono ou manobra em duas \
seções diferentes.
- GRAMÁTICA: concordância nominal obrigatória — adjetivos e artigos concordam em gênero e \
número com o substantivo que modificam. Exemplos corretos: "corrida emocionante" (não \
"corrida emocionantes"), "vitória dominante" (não "vitória dominantes"), "piloto veloz" \
(não "piloto velozes"). Substantivos femininos como "corrida", "vitória", "estratégia" \
exigem adjetivos no feminino singular.
- IDIOMA: escreva EXCLUSIVAMENTE em português do Brasil. Nunca use palavras em inglês \
(ex: "already", "race", "lap", "final lap") nem em italiano ou qualquer outra língua. \
Exceção permitida: nomes próprios de pilotos e equipes exatamente como no JSON."""

    contexto: str = dspy.InputField(
        desc=(
            "JSON com: `race` (vencedor, pódio, clima, volta mais rápida), "
            "`drivers` (perfil de cada piloto: grid, chegada, delta, compostos), "
            "`must_cover` (agenda: "
            "desenvolvimento=Safety Cars/abandonos/penalidades em ordem cronológica, "
            "analise_tecnica=manobras de pit mais decisivas com margens de tempo, "
            "destaques_individuais=pilotos a destacar com resultado). "
            "Cubra todos os itens de must_cover."
        )
    )

    titulo: str = dspy.OutputField(
        desc="Título jornalístico impactante. 1 linha, máx 12 palavras."
    )

    lead: str = dspy.OutputField(
        desc=(
            "2–3 frases, 40–60 palavras. Vencedor, resultado final e principal drama da corrida. "
            "Mencione condições climáticas de forma natural (sem temperatura). Seja direto e impactante."
        )
    )

    narrativa: str = dspy.OutputField(
        desc=(
            "2–3 parágrafos em prosa livre. Narre a corrida cronologicamente, integrando "
            "os acontecimentos táticos (manobras de pit, estratégias de pneu) naturalmente — "
            "sem criar seção separada para análise técnica. "
            "Cubra todos os itens de must_cover.desenvolvimento e must_cover.analise_tecnica. "
            "Cite voltas, margens de tempo e compostos de pneu onde relevante."
        )
    )

    destaques: str = dspy.OutputField(
        desc=(
            "1 parágrafo curto por piloto de must_cover.destaques_individuais. "
            "Para finalizadores (P1–P3): 2 frases sobre resultado e desempenho. "
            "Para DNFs: 1 frase direta com causa e volta do abandono. "
            "NÃO repita o que já está na narrativa. "
            "Total de parágrafos = total de pilotos na lista."
        )
    )

    conclusao: str = dspy.OutputField(
        desc="2 frases, 30–50 palavras. Contextualize o resultado no campeonato. Seja direto."
    )


# ---------------------------------------------------------------------------
# DSPy Module — Predict com campos separados, monta RelatorioSecoes no forward()
# ---------------------------------------------------------------------------

class GeradorRelatorio(dspy.Module):
    def __init__(self):
        # Predict com 6 OutputFields independentes: cada campo é gerado separadamente,
        # evitando que o modelo comprima tudo num único objeto JSON.
        self.generate = dspy.Predict(RelatorioJornalistico)

    def forward(self, contexto: str) -> dspy.Prediction:
        return self.generate(contexto=contexto)


# ---------------------------------------------------------------------------
# Token usage extraction from DSPy history
# ---------------------------------------------------------------------------

def _extract_token_usage(lm: dspy.LM) -> dict[str, int | float]:
    """Pull prompt/completion token counts and cost from all DSPy LM calls in this run."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cost_usd = 0.0

    try:
        for entry in lm.history:
            usage = entry.get("usage", {})
            prompt_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
            cost_usd += float(entry.get("cost") or 0.0)
    except Exception:
        pass

    if total_tokens == 0 and (prompt_tokens + completion_tokens) > 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


# ---------------------------------------------------------------------------
# LLM-C: Quality metrics
# ---------------------------------------------------------------------------

def _compute_hallucination_score(
    relatorio: RelatorioSecoes,
    race_summary: dict,
) -> float:
    """
    LLM-C: Penaliza menções a pilotos/equipes que não existem no race_summary.

    Verifica se códigos de 3 letras maiúsculas mencionados no relatório
    correspondem a pilotos reais da corrida.
    """
    # Coletar todos os drivers conhecidos da corrida
    known_drivers: set[str] = set()
    for p in race_summary.get("podium", []):
        known_drivers.add(p.get("driver", "").upper())
    for d in race_summary.get("dnfs", []):
        known_drivers.add(d.get("driver", "").upper())

    if not known_drivers:
        return 1.0  # Sem dados para verificar

    # Texto completo do relatório
    full_text = " ".join([
        relatorio.titulo,
        relatorio.lead,
        relatorio.narrativa,
        relatorio.destaques,
        relatorio.conclusao,
    ])

    # Extrair códigos de 3 letras maiúsculas entre parênteses (padrão do NarrativeContext)
    mentioned_codes = set(re.findall(r'\(([A-Z]{3})\)', full_text))

    if not mentioned_codes:
        return 1.0  # Sem códigos explícitos = sem alucinações detectadas

    invalid = mentioned_codes - known_drivers
    penalty = len(invalid) * 0.15
    return round(max(0.0, 1.0 - penalty), 3)


def _compute_event_coverage(
    relatorio: RelatorioSecoes,
    timeline: list | dict,
) -> float:
    """
    LLM-C / MTR-A: % de eventos da timeline mencionados no desenvolvimento e análise técnica.

    MTR-A: verifica se o sobrenome do piloto OU o número da volta aparece no texto.
    Isso corrige a métrica anterior que penalizava frases como "Alonso abandonou no
    meio da corrida" por não conter o número "34".

    Lógica: covered = lap_mentioned OR driver_name_mentioned (OR para cada driver field).
    """
    events = timeline if isinstance(timeline, list) else []
    if not events:
        return 1.0

    relevant_types = {"safety_car", "undercut", "overcut", "retirement", "penalty"}
    relevant = [e for e in events if e.get("type") in relevant_types]

    if not relevant:
        return 1.0

    # Texto completo dos campos narrativos
    text = " ".join([
        relatorio.narrativa,
        relatorio.destaques,
    ]).lower()

    covered = 0
    for evt in relevant:
        lap = evt.get("lap", 0)

        # 1. Número da volta mencionado
        lap_mentioned = f"volta {lap}" in text or f"lap {lap}" in text

        # 2. Sobrenome do(s) piloto(s) do evento mencionado
        driver_mentioned = False
        for field in ("driver", "winner", "loser"):
            code = str(evt.get(field, "")).upper()
            if not code:
                continue
            full_name = _DRIVER_NAMES.get(code, "")
            if full_name:
                # Verifica sobrenome (última palavra do nome completo)
                last_name = full_name.split()[-1].lower()
                if last_name in text:
                    driver_mentioned = True
                    break

        if lap_mentioned or driver_mentioned:
            covered += 1

    return round(covered / len(relevant), 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    race_summary: dict,
    timeline: dict | list,
    driver_profiles: dict | list,
    race_control: list | None = None,
    driver_quality_scores: dict | None = None,
    experiment_name: str = "F1_LLM_Reports",
    run_name: str | None = None,
) -> RelatorioSecoes:
    """
    Generate a structured journalistic F1 race report using DSPy + Groq.

    The LLM output is validated by RelatorioSecoes (Pydantic) before returning.

    Logs a full MLflow run with:
      - params: model, provider, DSPy module, race metadata
      - metrics: latency_seconds, tokens, tokens_per_second, hallucination_score,
                 event_coverage, word counts
      - artifacts: relatorio.json (structured), relatorio.txt (plain text)

    Returns:
        RelatorioSecoes — validated Pydantic object with all article sections.
    """
    from src.utils.config import config as _cfg

    lm = setup_lm()
    provider = _cfg.get("llm.provider", "groq")
    model_name = _cfg.get("llm.model", "llama-3.3-70b-versatile")

    # ── MLflow setup ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(_cfg.get_mlflow_tracking_uri())
    mlflow.set_experiment(experiment_name)

    try:
        mlflow.dspy.autolog()
    except (AttributeError, RuntimeError):
        # AttributeError: mlflow version sem suporte a dspy.autolog
        # RuntimeError: dspy.settings thread-ownership restriction (ex: Streamlit)
        pass

    year = race_summary.get("year", "unknown")
    round_num = race_summary.get("round", "unknown")
    if run_name is None:
        run_name = (
            f"report_y{year}_r{round_num:02d}"
            if isinstance(round_num, int)
            else f"report_{year}_{round_num}"
        )

    # ── Build compact NarrativeContext (~60% fewer tokens than raw JSONs) ────
    timeline_list = timeline if isinstance(timeline, list) else []
    profiles_list = driver_profiles if isinstance(driver_profiles, list) else []

    context_str = build_narrative_context(
        race_summary=race_summary,
        timeline=timeline_list,
        driver_profiles=profiles_list,
        race_control=race_control,
        driver_quality_scores=driver_quality_scores,
    )
    input_chars = len(context_str)

    # Clear history so usage extraction only captures this run
    lm.history.clear()

    # ── MLflow run ───────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": model_name,
            "provider": provider,
            "dspy_module": "Predict",
            "year": year,
            "round": round_num,
            "winner": race_summary.get("winner", ""),
        })
        mlflow.log_metric("input_chars", input_chars)

        # ── DSPy call with timing ────────────────────────────────────────────
        # Use dspy.context() instead of dspy.configure() so the LM is set as a
        # thread-local override — safe to call from any thread (e.g. Streamlit).
        retry_count = 0
        t0 = time.perf_counter()
        generator = GeradorRelatorio()

        with dspy.context(lm=lm):
            try:
                pred = generator(contexto=context_str)
            except Exception:
                retry_count += 1
                pred = generator(contexto=context_str)

        latency_s = time.perf_counter() - t0

        # LLM-C: ttft_seconds — Time To First Token
        # DSPy não expõe TTFT diretamente; tenta extrair de history se disponível,
        # caso contrário registra 0 (indica que TTFT não foi medido).
        ttft_seconds = 0.0
        try:
            for entry in lm.history:
                ttft = entry.get("time_to_first_token") or entry.get("ttft") or 0.0
                if ttft:
                    ttft_seconds = float(ttft)
                    break
        except Exception:
            pass

        # Monta RelatorioSecoes a partir dos campos do Prediction
        relatorio = RelatorioSecoes(
            titulo=pred.titulo,
            lead=pred.lead,
            narrativa=pred.narrativa,
            destaques=pred.destaques,
            conclusao=pred.conclusao,
        )

        # ── Token usage & derived metrics ────────────────────────────────────
        usage = _extract_token_usage(lm)
        cost_usd = usage.pop("cost_usd")

        # LLM-C: Derived quality metrics
        tokens_per_second = (
            round(usage["completion_tokens"] / latency_s, 1)
            if latency_s > 0 and usage["completion_tokens"] > 0
            else 0.0
        )
        hallucination_score = _compute_hallucination_score(relatorio, race_summary)
        event_coverage = _compute_event_coverage(relatorio, timeline_list)

        mlflow.log_metrics({
            "latency_seconds": round(latency_s, 3),
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "cost_usd": cost_usd,
            # LLM-C: new metrics
            "ttft_seconds": round(ttft_seconds, 3),
            "tokens_per_second": tokens_per_second,
            "retry_count": retry_count,
            "hallucination_score": hallucination_score,
            "event_coverage": event_coverage,
            # Word counts
            "word_count_total": relatorio.word_count(),
            "word_count_titulo": len(relatorio.titulo.split()),
            "word_count_lead": len(relatorio.lead.split()),
            "word_count_narrativa": len(relatorio.narrativa.split()),
            "word_count_destaques": len(relatorio.destaques.split()),
            "word_count_conclusao": len(relatorio.conclusao.split()),
        })

        if usage["total_tokens"] > 0 and cost_usd > 0:
            mlflow.log_metric(
                "cost_per_1k_tokens",
                round(cost_usd / usage["total_tokens"] * 1000, 6),
            )

        mlflow.set_tags({
            "winner": race_summary.get("winner", ""),
            "type": "llm_report",
            "safety_car_count": str(race_summary.get("safety_car_count", "")),
        })

        # ── Artifacts ────────────────────────────────────────────────────────
        mlflow.log_text(relatorio.model_dump_json(indent=2), "relatorio.json")
        mlflow.log_text(relatorio.to_text(), "relatorio.txt")

        _print_run_summary(run_name, latency_s, usage, cost_usd, relatorio, tokens_per_second, hallucination_score, event_coverage, ttft_seconds)

    return relatorio


def _print_run_summary(
    run_name: str,
    latency_s: float,
    usage: dict[str, int],
    cost_usd: float,
    relatorio: RelatorioSecoes,
    tokens_per_second: float = 0.0,
    hallucination_score: float = 1.0,
    event_coverage: float = 1.0,
    ttft_seconds: float = 0.0,
) -> None:
    print(f"\n[MLflow] Run: {run_name}")
    print(f"  latency        : {latency_s:.2f}s")
    print(f"  ttft           : {ttft_seconds:.3f}s")
    print(f"  tokens/s       : {tokens_per_second:.0f}")
    print(f"  prompt tokens  : {usage['prompt_tokens']}")
    print(f"  output tokens  : {usage['completion_tokens']}")
    print(f"  total tokens   : {usage['total_tokens']}")
    print(f"  total words    : {relatorio.word_count()}")
    print(f"  cost (USD)     : ${cost_usd:.6f}")
    print(f"  hallucination  : {hallucination_score:.3f}")
    print(f"  event_coverage : {event_coverage:.3f}")
