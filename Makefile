.PHONY: figures gate test m3-diagram diagnostics diagnostics-smoke tier1-matrix tier1-dry

gate:
	python3 scripts/validation_gate.py

test:
	python3 -m pytest tests/ -q

figures: m3-diagram

m3-diagram:
	python3 scripts/make_phase_diagram.py --index artifacts/runs/ippo/m3/index.json --out-dir artifacts/runs/ippo/m3

# Eval-only root-cause diagnostics (no training). Writes analysis/DIAGNOSTICS.md
# and analysis/figs/diag/. Uses frozen M3 checkpoints + existing logs.
diagnostics:
	python3 -m analysis.diag.run_all --episodes-d2 100 --episodes-d5 30

diagnostics-smoke:
	python3 -m analysis.diag.run_all --max-seeds 1 --episodes-d2 5 --episodes-d4 5 --episodes-d5 5 --force

# Tier-1 v1.1 parallel matrix (Regime A/B/C × topo × cap × demand). Resumable.
tier1-dry:
	python3 scripts/run_tier1_matrix.py --dry-run

tier1-matrix:
	python3 scripts/run_tier1_matrix.py --workers 8 --n-envs 64 --skip-existing
