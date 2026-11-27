.PHONY: setup lint typecheck test smoke experiments report docker-build verify clean help

PYTHON ?= python3
PIP ?= pip3

help:
	@echo "swarm-isr-sar-sim — targets principais"
	@echo "  setup          Instala + deps de desenvolvimento"
	@echo "  lint           ruff check + ruff format --check"
	@echo "  typecheck      mypy strict no core"
	@echo "  test           pytest + cobertura"
	@echo "  smoke          Cenário S1 (indisponível até o Lote 5)"
	@echo "  experiments    Protocolo S1–S7 (indisponível até o Lote 7)"
	@echo "  report         Relatório (indisponível até o Lote 6)"
	@echo "  docker-build   Tier B (indisponível até o Lote 9)"
	@echo "  verify         Verificação completa (indisponível até o Lote 11)"
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
	pytest --cov=swarm_core --cov-report=term-missing --cov-fail-under=80

smoke:
	@if [ -f src/swarm_core/cli.py ]; then \
		swarm-sim run --scenario S1 --seed 0 --profile clean --time-limit 60 --output results/smoke --quiet; \
	else \
		echo "BLOQUEADO: CLI ainda não implementada (Lote 5). Nenhum smoke test foi executado."; \
		exit 2; \
	fi

experiments:
	@if [ -f analysis/run_experiments.py ]; then \
		$(PYTHON) -m analysis.run_experiments --seeds 0-29 --scenarios S1,S2,S3,S4,S5,S6,S7; \
	else \
		echo "BLOQUEADO: runner de experimentos ainda não implementado (Lote 7)."; \
		exit 2; \
	fi

report:
	@if [ -f src/swarm_core/cli.py ]; then \
		swarm-sim report --run-dir results/smoke --output report.html; \
	else \
		echo "BLOQUEADO: CLI/relatórios ainda não implementados (Lote 6)."; \
		exit 2; \
	fi

docker-build:
	@if [ -f docker/docker-compose.yml ]; then \
		docker compose -f docker/docker-compose.yml build; \
	else \
		echo "BLOQUEADO: configuração Docker Tier B ainda não implementada (Lote 9)."; \
		exit 2; \
	fi

verify:
	@if [ -f scripts/verify_all.py ]; then \
		$(PYTHON) scripts/verify_all.py; \
	else \
		echo "BLOQUEADO: verificador completo ainda não implementado (Lote 11)."; \
		exit 2; \
	fi

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
