# Itens não verificados empiricamente (VERIFY:)

Este arquivo lista decisões de versão, flags, tópicos e APIs que foram escolhidas com base em documentação pública, mas **ainda não foram executadas** neste repositório.

| Item | Escolha atual | Fonte / motivo | Status |
|------|---------------|----------------|--------|
| ROS 2 distro | Humble | LTS + ampla adoção industrial | NÃO EXECUTADO |
| Gazebo | Harmonic + ros_gz | Compatibilidade declarada com Humble (VERIFY: docs oficiais) | NÃO EXECUTADO |
| PX4-Autopilot | tag v1.15.x | Estável recente; bridge uXRCE-DDS | NÃO EXECUTADO |
| Micro XRCE-DDS Agent | versão correspondente à branch px4_msgs | Documentação PX4 | NÃO EXECUTADO |
| Modelo Gazebo | gz_x500 com câmera downward | Padrão PX4 SITL multi-vehicle | NÃO EXECUTADO |
| Física / RTF | 250 Hz, RTF-alvo ≥ 0.5 com 10 drones | Estimativa de perfil lite | NÃO EXECUTADO |
| QoS ROS 2 | RELIABLE depth 10 para comandos; BEST_EFFORT depth 5 para visão | Prática comum em swarm | NÃO EXECUTADO |
| ArUco detector | OpenCV padrão | Baseline estável para calibração s/f | NÃO EXECUTADO |
| YOLO nano/ONNX | Opcional | Alternativa futura; não no caminho crítico | NÃO EXECUTADO |

Atualize este arquivo sempre que um item for executado e validado (passe o status para EXECUTADO-PASSOU ou EXECUTADO-FALHOU + observação).
