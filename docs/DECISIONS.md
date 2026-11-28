# Architecture Decision Records (ADR)

Formato curto: Contexto → Decisão → Alternativas → Consequência.

## ADR-001 — Duas camadas (Tier A / Tier B)

**Contexto**: Necessidade de critérios quantitativos de aceite medidos de forma determinística e rápida em CI, ao mesmo tempo em que se deseja demonstração realista com PX4 SITL + Gazebo.

**Decisão**: Tier A = Python puro (numpy/scipy/pydantic) contendo *toda* a lógica de decisão. Tier B = adaptadores finos ROS 2 que apenas traduzem mensagens ↔ estruturas do core.

**Alternativas**: (1) Tudo em ROS 2 desde o início; (2) Simulação monólito em Gazebo.

**Consequência**: Testes de paridade obrigatórios. Falha do Tier B nunca bloqueia entrega do Tier A. Smoke e experimentos rodam em segundos/minutos no Colab/Codespaces.

## ADR-002 — Determinismo estrito

**Contexto**: Reprodutibilidade científica e debugging de comportamentos emergentes do enxame.

**Decisão**: `dt = 0.1 s` fixo, processamento por ID crescente, `SeedSequence.spawn` por componente, zero chamada a relógio de parede ou `random` global. Teste de hash SHA-256 do log de eventos.

**Alternativas**: Relógio de parede + seeds parciais.

**Consequência**: Duas execuções com a mesma seed produzem logs idênticos. Facilita testes de propriedade com Hypothesis.

## ADR-003 — Alocação padrão CBBA

**Contexto**: Necessidade de alocação distribuída robusta a partições e perda de pacotes.

**Decisão**: CBBA (Choi, Brunet & How, 2009) como algoritmo principal; auction e greedy como baselines; fallback automático para auction se CBBA não convergir em 10 s.

**Alternativas**: Somente leilão centralizado; market-based puro.

**Consequência**: Bundle size máximo L_t = 4. Tabela de resolução de conflitos implementada explicitamente. Testes de ausência de duplicatas após convergência.

## ADR-004 — Consenso raft_like + bully

**Contexto**: Failover de líder ≤ 10 s (O5) sob perda de pacotes.

**Decisão**: `raft_like` (termo, voto, heartbeat 1 Hz, timeout seeded 3.0–4.5 s) como padrão; `bully` (menor ID) como alternativa configurável. Em partição cada componente elege degraded_leader.

**Alternativas**: Somente bully; Paxos completo.

**Consequência**: Orçamento de tempo: detecção ≤ 3 s + eleição ≤ 3 s + sync ≤ 2 s. Testes com 100 seeds.

## ADR-005 — Log auditável com hash chain

**Contexto**: Rastreabilidade de comandos do operador e eventos de missão (requisito de pesquisa/ética).

**Decisão**: JSONL append-only; cada linha contém `prev_sha256`. Função `verify_audit_log()` valida a cadeia inteira.

**Alternativas**: Banco de dados com assinaturas; log simples sem integridade.

**Consequência**: Adulteração é detectável. Modo sombra registra mas não aplica comandos.

## ADR-006 — Versões ROS 2 / Gazebo / PX4

**Contexto**: Incompatibilidades frequentes entre stacks.

**Decisão**: Fixar em `docker/versions.env` e `docs/VERSIONS.md`. Ponto de partida: Ubuntu 22.04 + ROS 2 Humble + Gazebo Harmonic + PX4 v1.15.x. Tudo marcado VERIFY: e listado em UNVERIFIED.md até validação empírica.

**Consequência**: Tier A permanece independente. CI do Tier B só faz build + lint + launch_testing leve.

## ADR-007 — Um único formatador (ruff)

**Contexto**: Evitar conflito entre black e ruff-format.

**Decisão**: Usar exclusivamente `ruff format` (e `ruff check`). Removido black do pipeline.

**Consequência**: `make lint` e pre-commit usam apenas ruff. Configuração centralizada em pyproject.toml.


## ADR-008 — Sincronização Gossip por watermark confirmado

**Contexto**: O flood epidêmico reenviava todo o histórico para todos os peers, sem registrar o progresso individual de cada destinatário. Isso gerava tráfego redundante e não correspondia ao contrato documentado de sincronização a partir do watermark do receptor.

**Decisão**: Cada nó mantém watermarks de aplicação confirmados por peer e origem. Ao receber um lote, o destinatário envia uma confirmação com seus watermarks contíguos; o emissor retransmite apenas lotes posteriores ao watermark confirmado. Lotes da própria origem do peer não são enviados de volta a ele. Se o lote ou a confirmação se perder, o watermark não avança e o lote permanece elegível para retransmissão.

**Alternativas**: Flood completo em todo ciclo; estado de sincronização centralizado fora dos nós; ACK por lote individual.

**Consequência**: O mecanismo reduz retransmissões redundantes e recupera perdas por repetição idempotente. O modelo de ACK é simulado pelo NetworkSim e não representa ainda um protocolo de transporte distribuído em produção. Testes devem cobrir progresso independente por peer, perda de lote, perda de ACK, partições e convergência.


## ADR-009 — Dual-World state contracts (P2.0)

**Contexto**: O baseline mistura estado físico verdadeiro, estado estimado do agente e crenças probabilísticas locais. Essa mistura permite que código de agente ou planejador leia acidentalmente a posição verdadeira do drone ou o objeto `World`.

**Decisão**: Introduzir `PhysicalTruthState`, `AgentLocalState` e `BeliefSnapshot` como dataclasses congeladas, com vetores e coleções representados por tuplas imutáveis. `PhysicalTruthState` é reservado ao núcleo físico e à avaliação; agentes e planejadores devem consumir `AgentLocalState` e `BeliefSnapshot`. Testes AST bloqueiam novos imports/referências ao mundo físico e acessos diretos a `drone.position`.

**Allowlist temporária**: `mission.py`, `drone.py`, `sensors.py` e `planning/boustrophedon.py` permanecem legados. Os três primeiros serão migrados em P2.1/P2.2; o planejador de cobertura será migrado em P2.2. A allowlist é explícita e deve ser reduzida, nunca ampliada sem justificativa documentada.

**Consequência**: P2.0 cria contratos e barreiras de regressão sem alterar a dinâmica, o fluxo de observações, o Gossip ou o EventLog. A existência dos contratos não significa que a arquitetura Dual-World já esteja integralmente migrada.
