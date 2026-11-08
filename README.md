# Agent DJ — Multi-Agent Interface Prototype

A Streamlit interface prototype for exploring a DJ workflow organized around three conceptual agents.

## Current interface

The current Streamlit app presents three roles:

1. **Music selection** — represents the selection of tracks for a session context.
2. **Mixing** — represents the mixing stage.
3. **Audience analysis** — represents audience-oriented analysis.

Users can choose a session context and enter a track for a demonstration.

## Important implementation note

The current interface is a demonstrator. The displayed agent activity, session completion messages, and example scores are simulated in the Streamlit UI. They should not be interpreted as evidence that three autonomous agents or real-time audio analysis are already running.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Repository structure

- `streamlit_app.py` — current Streamlit demonstration interface.
- `requirements.txt` — Python dependencies for the project.
- `MON_DJ_IA_COMPLET/` — additional project materials.

## Roadmap

- Connect the interface to actual agent implementations.
- Replace simulated metrics with measurable outputs.
- Add tests, architecture documentation, and reproducible examples.

## Related work

This repository is intentionally separate from the independent intelligent DJ selection prototype. The two projects are not being merged.
