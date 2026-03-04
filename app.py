"""
PitWall AI — Streamlit Interface

Interface web para o pipeline F1. Permite selecionar ano e corrida,
executar o pipeline com feedback visual por fase, e exibir o relatório
gerado (com cache automático se já existir em disco).

Uso:
    uv run streamlit run app.py
"""

import json
import sys
from pathlib import Path

import fastf1
import streamlit as st

# Garante que imports `from src.xxx` funcionam ao rodar da raiz do projeto
sys.path.insert(0, str(Path(__file__).parent))

from cli.pipeline_steps.extraction import run_extraction_phase
from cli.pipeline_steps.preprocessing import run_preprocessing_phase
from cli.pipeline_steps.ml import run_ml_phase
from cli.pipeline_steps.events import run_events_phase
from cli.pipeline_steps.llm import run_llm_phase
from src.models.relatorio import RelatorioSecoes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_available_years() -> list[int]:
    """Anos com dados históricos disponíveis (2026 excluído — temporada não encerrada)."""
    return list(range(2025, 2017, -1))  # 2025..2018 decrescente


@st.cache_data(show_spinner=False)
def get_race_schedule(year: int) -> list[tuple[int, str]]:
    """
    Retorna lista de (RoundNumber, EventName) para o calendário do ano.
    Filtra entradas de pré-temporada/testes (EventFormat == 'testing').
    Resultado cacheado pelo Streamlit para evitar chamadas repetidas à API.
    """
    cache_dir = Path.home() / ".cache" / "fastf1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    races = []
    for _, row in schedule.iterrows():
        if str(row.get("EventFormat", "")).lower() == "testing":
            continue
        round_num = int(row["RoundNumber"])
        name = str(row["EventName"])
        races.append((round_num, name))
    return sorted(races, key=lambda x: x[0])


def _report_path(year: int, round_num: int) -> Path:
    return Path("data/timelines/races") / f"{year}" / f"round_{round_num:02d}" / "relatorio.json"


def check_report_exists(year: int, round_num: int) -> bool:
    return _report_path(year, round_num).exists()


def load_existing_report(year: int, round_num: int) -> RelatorioSecoes:
    data = json.loads(_report_path(year, round_num).read_text(encoding="utf-8"))
    return RelatorioSecoes(**data)


def run_pipeline_with_progress(year: int, round_num: int) -> RelatorioSecoes:
    """Executa as 5 fases do pipeline com st.status() por fase."""

    with st.status("Fase 1: Extração de dados...", expanded=True) as s:
        race_dir = run_extraction_phase(year=year, round_num=round_num)
        s.update(label="Fase 1: Extração concluída", state="complete", expanded=False)

    with st.status("Fase 2: Pré-processamento...", expanded=True) as s:
        processed_dir = run_preprocessing_phase(
            race_dir=race_dir,
            year=year,
            round_num=round_num,
        )
        s.update(label="Fase 2: Pré-processamento concluído", state="complete", expanded=False)

    with st.status("Fase 3: Machine Learning...", expanded=True) as s:
        ml_dir = run_ml_phase(
            processed_dir=processed_dir,
            year=year,
            round_num=round_num,
        )
        s.update(label="Fase 3: Machine Learning concluído", state="complete", expanded=False)

    with st.status("Fase 4: Eventos estruturados...", expanded=True) as s:
        timeline_dir = run_events_phase(
            ml_dir=ml_dir,
            processed_dir=processed_dir,
            year=year,
            round_num=round_num,
        )
        s.update(label="Fase 4: Eventos estruturados concluídos", state="complete", expanded=False)

    with st.status("Fase 5: Relatório LLM (Groq)...", expanded=True) as s:
        run_llm_phase(
            timeline_dir=timeline_dir,
            year=year,
            round_num=round_num,
            processed_dir=processed_dir,
        )
        s.update(label="Fase 5: Relatório gerado", state="complete", expanded=False)

    return load_existing_report(year, round_num)


def display_report(relatorio: RelatorioSecoes, year: int, round_num: int) -> None:
    """Renderiza o relatório em Markdown e oferece botão de download."""
    md = relatorio.to_markdown()
    st.markdown(md)
    st.download_button(
        label="Baixar Markdown",
        data=md,
        file_name=f"relatorio_{year}_r{round_num:02d}.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PitWall AI",
    page_icon="🏁",
    layout="wide",
)

# Sidebar — seletores
with st.sidebar:
    st.title("PitWall AI")
    st.caption("Pipeline F1: dados → ML → relatório jornalístico")

    year = st.selectbox("Ano", get_available_years(), index=0)

    with st.spinner("Carregando calendário..."):
        races = get_race_schedule(year)

    if not races:
        st.error("Calendário não encontrado para este ano.")
        st.stop()

    race_options = {f"R{rn:02d} — {name}": rn for rn, name in races}
    selected_label = st.selectbox("Corrida", list(race_options.keys()))
    round_num = race_options[selected_label]

# Main area
_, name = next((r for r in races if r[0] == round_num), (round_num, selected_label))
st.header(f"{name} {year}")

report_exists = check_report_exists(year, round_num)

# Estado de sessão para controlar regeneração forçada
if "force_regenerate" not in st.session_state:
    st.session_state.force_regenerate = False

if report_exists and not st.session_state.force_regenerate:
    st.success("Relatório disponível em disco.")
    col1, col2 = st.columns([1, 1])

    with col1:
        ver = st.button("Ver Relatório", type="primary")
    with col2:
        regen = st.button("Regerar")

    if regen:
        st.session_state.force_regenerate = True
        st.rerun()

    if ver or "show_report" in st.session_state:
        if ver:
            st.session_state.show_report = True
        relatorio = load_existing_report(year, round_num)
        display_report(relatorio, year, round_num)

else:
    if st.session_state.force_regenerate:
        st.info("Regenerando relatório...")
    else:
        st.info("Relatório ainda não gerado para esta corrida.")

    if st.button("Gerar Relatório", type="primary") or st.session_state.force_regenerate:
        st.session_state.force_regenerate = False
        st.session_state.pop("show_report", None)
        try:
            relatorio = run_pipeline_with_progress(year, round_num)
            st.success("Pipeline concluído com sucesso!")
            display_report(relatorio, year, round_num)
        except Exception as exc:
            st.error(f"Erro durante o pipeline: {exc}")
            with st.expander("Detalhes do erro"):
                import traceback
                st.code(traceback.format_exc())
