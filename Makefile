.PHONY: setup lint typecheck test smoke experiments report docker-build verify clean help

PYTHON ?= python3
PIP ?= pip3

help:
	@echo "swarm-isr-sar-sim — targets principais"
	@echo "  setup          Instala o pacote em modo editável + deps de desenvolvimento"
	@echo "  lint           ruff check + ruff format --check"
	@echo "  typecheck      mypy strict no core"
	@echo "  test           pytest + cobertura"
	@echo "  smoke          Cenário S1 seed 0 (rápido) → results/smoke"
	@echo "  experiments    Roda protocolo completo de experimentos"
	@echo "  report         Gera report.html a partir do último run"
	@echo "  docker-build   Build das imagens Tier B (sem Gazebo)"
	@echo "  verify         scripts/verify_all.py → docs/VERIFICATION.md"
	@echo "  clean          Remove caches e artefatos gerados"

setup:
	$(PIP) install -e ".[dev,report,experiments]"
	pre-commit install || true

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy src/swarm_core

test:
	pytest --cov=swarm_core --cov-report=term-missing --cov-fail-under=0

smoke:
	@mkdir -p results/smoke
	@if [ -f src/swarm_core/cli.py ]; then \
		swarm-sim run --scenario S1 --seed 0 --profile clean --time-limit 60 --output results/smoke --quiet; \
		echo "Smoke OK (S1 seed 0) → results/smoke"; \
	else \
		echo "NÃO EXECUTADO: src/swarm_core/cli.py ainda não existe (Lote 5)"; \
		exit 0; \
	fi

experiments:
	@if [ -f analysis/run_experiments.py ]; then \
		$(PYTHON) -m analysis.run_experiments --seeds 0-29 --scenarios S1,S2,S3,S4,S5,S6,S7; \
	else \
		echo "NÃO EXECUTADO: analysis/run_experiments.py ainda não existe (Lote 7)"; \
	fi

report:
	@if [ -f src/swarm_core/cli.py ]; then \
		swarm-sim report --run-dir results/smoke --output report.html; \
	else \
		echo "NÃO EXECUTADO: cli ainda não existe"; \
	fi

docker-build:
	@if [ -f docker/docker-compose.yml ]; then \
		docker compose -f docker/docker-compose.yml build; \
	else \
		echo "NÃO EXECUTADO: docker/docker-compose.yml ainda não existe (Lote 9)"; \
		exit 0; \
	fi

verify:
	@if [ -f scripts/verify_all.py ]; then \
		$(PYTHON) scripts/verify_all.py; \
	else \
		echo "NÃO EXECUTADO: scripts/verify_all.py ainda não existe (Lote 11)"; \
	fi

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
