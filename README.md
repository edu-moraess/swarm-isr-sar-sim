# swarm-isr-sar-sim

**Enxame de drones para ISR e Search-and-Rescue (SAR) em simulação**

> Portfólio avançado · Engenharia de Computação / Robótica / Sistemas Distribuídos  
> Nível: TCC / pesquisa aplicada · Licença MIT · Python 3.11+

[![CI](https://github.com/edu-moraess/swarm-isr-sar-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/edu-moraess/swarm-isr-sar-sim/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Visão geral

Sistema de enxame de drones **estritamente simulado** para missões de:

- **SAR** (Search-and-Rescue): cobertura bayesiana de área, detecção e confirmação de alvos sintéticos.
- **ISR** (Intelligence, Surveillance, Reconnaissance): setores com prioridade, minimização de staleness, relay e alertas de eventos.

Arquitetura em **duas camadas**:

| Camada | Tecnologia | Responsabilidade |
|--------|------------|------------------|
| **Tier A** | Python puro (numpy, scipy, pydantic) | Toda a lógica: crença, planejamento, alocação (CBBA), consenso, rede, FSM, métricas |
| **Tier B** | ROS 2 Humble + Gazebo Harmonic + PX4 SITL | Adaptadores finos, mundos, launch, impairments, dashboard Foxglove |

Critérios quantitativos de aceite (O1–O6) são medidos **exclusivamente** no Tier A.

## Quickstart (≤ 5 comandos)

```bash
git clone https://github.com/edu-moraess/swarm-isr-sar-sim.git
cd swarm-isr-sar-sim
make setup
make smoke          # S1 seed 0 → results/smoke
swarm-sim report --run-dir results/smoke --output report.html
```

Abra `report.html` no navegador (funciona no celular).

## Objetivos mensuráveis

| ID | Objetivo | Medida |
|----|----------|--------|
| O1 | Cobertura ≥ 90 % em ≤ 15 min, ≥ 10 drones, 200×200 m | % células livres com ≥ 1 observação válida |
| O2 | ≥ 90 % alvos detectados, FP ≤ 5 % | detecção / total; FP em clutter |
| O3 | Perda de até 3 nós → re-tarefa ≤ 30 s | latência do timeout de heartbeat à nova atribuição |
| O4 | 20 % perda + 200 ms latência | cobertura ≥ 90 %, sem tarefa duplicada > 5 s |
| O5 | Failover de líder ≤ 10 s | até SwarmState do novo líder aceito por ≥ 80 % dos nós vivos |
| O6 | Relatório com métricas + 2+ baselines | 30 seeds, IC 95 %, teste estatístico |

## Estrutura do repositório

```
swarm-isr-sar-sim/
├── src/swarm_core/          # Tier A (toda a lógica)
├── tests/                   # pytest + hypothesis
├── config/                  # mission.yaml, swarm.yaml, …
├── analysis/                # experimentos e geração de EXPERIMENTS.md
├── docker/                  # Tier B (a partir do Lote 9)
├── worlds/                  # SDF Gazebo
├── docs/                    # DECISIONS, ETHICS, UNVERIFIED, ARCHITECTURE…
├── scripts/                 # materialize, verify_all, git_init
├── notebooks/               # Colab quickstart
└── results/                 # CSVs, figuras, reports
```

## Comandos principais

```bash
make setup          # instala + pre-commit
make lint           # ruff check + ruff format --check
make typecheck      # mypy strict
make test           # pytest + cobertura
make smoke          # S1 rápido → results/smoke
make experiments    # protocolo completo (S1–S7 × 30 seeds)
make report         # gera report.html
make verify         # scripts/verify_all.py → docs/VERIFICATION.md
```

## Ética e escopo

Ambiente **estritamente simulado**, educacional e de pesquisa.  
Dados sintéticos. Sem instruções de armamento, targeting real ou identificação de pessoas.  
Veja [docs/ETHICS.md](docs/ETHICS.md).

## Status de desenvolvimento

| Fase | Lote | Status |
|------|------|--------|
| Scaffolding | 0 | Completo |
| Core primitives | 1 | — |
| Planning + FSM | 2 | — |
| Network + Consensus | 3 | — |
| Allocation + Retask | 4 | — |
| Sim + CLI + Smoke | 5 | — |
| Operator + Report | 6 | — |
| Experimentos | 7 | — |
| Tier B interfaces | 8 | — |
| Docker / PX4 / Gazebo | 9 | — |
| Documentação final | 10 | — |
| Verificação completa | 11 | — |

## Licença

MIT © 2026 Eduardo Moraes
