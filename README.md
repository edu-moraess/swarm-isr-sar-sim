# swarm-isr-sar-sim

**Simulador multiagente de drones para Search-and-Rescue (SAR) e Intelligence, Surveillance and Reconnaissance (ISR).**

> Pesquisa aplicada em sistemas distribuídos, robótica e inteligência multiagente.  
> Python 3.11+ · Licença MIT

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Visão geral

O projeto implementa um ambiente **estritamente simulado** para estudar coordenação, percepção e compartilhamento de informação entre agentes autônomos.

- **SAR:** cobertura de área, observações e atualização probabilística de crença.
- **ISR:** exploração por setores, compartilhamento de observações e eventos.
- **Sistemas distribuídos:** comunicação sujeita a latência, perda de pacotes e partições.

A arquitetura está organizada em duas camadas planejadas:

| Camada | Responsabilidade |
|---|---|
| **Tier A — Python** | Núcleo de simulação, crença, planejamento, rede, FSM, energia e coordenação. |
| **Tier B — ROS 2 / Gazebo / PX4** | Integração robótica e simulação física, ainda não disponível neste snapshot. |

Os objetivos quantitativos O1–O6 são metas de pesquisa. Eles não devem ser interpretados como resultados já demonstrados.

## Estado executável atual

O núcleo Python e seus testes automatizados estão presentes. O MissionLoop v0 oferece uma orquestração determinística mínima de uma missão sintética no Tier A. A CLI, o runner configurável de cenários, a geração de relatórios e o runner de experimentos ainda não foram implementados. Por isso, os comandos swarm-sim, make smoke, make report e make experiments não são apresentados como funcionalidades disponíveis.

## Quickstart — validação do núcleo

\`\`\`bash
git clone https://github.com/edu-moraess/swarm-isr-sar-sim.git
cd swarm-isr-sar-sim
python -m pip install -e ".[dev,report,experiments]"
pytest
\`\`\`

Validações adicionais:

\`\`\`bash
ruff check src tests
ruff format --check src tests
mypy src/swarm_core
\`\`\`

No Colab, os mesmos comandos podem ser executados após clonar o repositório. O projeto não exige ROS 2, Gazebo ou PX4 para testar o Tier A.

## Objetivos mensuráveis planejados

| ID | Objetivo | Medida proposta |
|---|---|---|
| O1 | Cobertura de área | ≥ 90% em até 15 min, com ≥ 10 drones em área 200 × 200 m |
| O2 | Detecção de alvos sintéticos | ≥ 90% de detecção e ≤ 5% de falsos positivos |
| O3 | Retarefa após perda de nós | Até 30 s após detecção da falha |
| O4 | Robustez de rede | Avaliação com 20% de perda e 200 ms de latência |
| O5 | Failover de liderança | Até 10 s, conforme critério distribuído formalizado |
| O6 | Reprodutibilidade experimental | 30 seeds, intervalos de confiança e baselines |

Essas metas ainda dependem da implementação dos módulos de alocação, consenso, execução de cenários e protocolo experimental.

## Estrutura presente

\`\`\`text
src/swarm_core/    Núcleo Python
tests/             Testes automatizados
config/            Configurações YAML
docs/              Decisões arquiteturais e escopo ético
scripts/           Ferramentas de materialização
results/           Diretório reservado para resultados
\`\`\`

Diretórios de experimentos, integração ROS 2/Gazebo/PX4 e relatórios serão adicionados quando seus respectivos módulos forem implementados.

## Ética e escopo

Ambiente estritamente simulado, educacional e de pesquisa. O escopo atual utiliza ambientes e observações sintéticos; não contempla targeting real, armamento ou identificação de pessoas.

Consulte [docs/ETHICS.md](docs/ETHICS.md) e [docs/UNVERIFIED.md](docs/UNVERIFIED.md).

## Roadmap de implementação

| Lote | Escopo | Situação |
|---|---|---|
| 0 | Scaffolding | Presente |
| 1–2 | Primitivas, planejamento e FSM | Parcialmente implementados |
| 3 | Rede e inteligência distribuída | Em consolidação |
| 4 | Alocação e retarefa | Pendente |
| 5 | Orquestração, CLI e smoke test | Pendente |
| 6 | Interface de operação e relatórios | Pendente |
| 7 | Experimentos reproduzíveis | Pendente |
| 8–9 | Interfaces ROS 2, Gazebo e PX4 | Pendente |
| 10–11 | Documentação e verificação integrada | Pendente |

## Licença

MIT © 2026 Eduardo Moraes
