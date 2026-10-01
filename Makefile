PY=./.venv/bin/python
ADFD=./.venv/bin/adfd

install:
	$(PY) -m pip install -e .

bootstrap:
	$(ADFD) bootstrap --n 120

smoke: bootstrap
	$(ADFD) run --detector reference
	$(ADFD) sweep --detector reference

test:
	$(PY) -m pytest -q

.PHONY: install bootstrap smoke test
