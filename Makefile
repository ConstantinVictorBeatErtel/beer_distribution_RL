.PHONY: figures gate test m3-diagram

gate:
	python scripts/validation_gate.py

test:
	python -m pytest tests/ -q

figures: m3-diagram

m3-diagram:
	python scripts/make_phase_diagram.py --index artifacts/runs/ippo/m3/index.json --out-dir artifacts/runs/ippo/m3
