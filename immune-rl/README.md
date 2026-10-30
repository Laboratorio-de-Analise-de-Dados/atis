# immune-rl

Simulação do sistema imunológico como um grafo direcionado, com aprendizado por
reforço multiagente. O ambiente é um mundo emergente: começa com um único
macrófago em repouso e a resposta imune (reconhecimento, secreção de mediadores,
recrutamento, inflamação, dano e reparo) surge passo a passo.

## Modelo biológico

| Nó | Papel |
|---|---|
| `PAMP` | Sinal de perigo derivado do patógeno, reconhecido pelo macrófago |
| `Pathogen` | Carga do patógeno |
| `Macrophage` | Célula com ativação, memória e fenótipo M0/M1/M2 |
| `ProCytokine` | TNF-α/IL-6: pró-inflamatória, ativa macrófagos já ativados |
| `AntiCytokine` | IL-10: inibe ativação e produção pró-inflamatória |
| `Chemokine` | CCL2: recruta novos macrófagos |
| `RepairFactor` | TGF-β: repara dano tecidual |
| `TissueDamage` | Dano acumulado por patógeno e inflamação |

O fenótipo é emergente: cada macrófago mantém `pro_score` e `anti_score` (média
móvel das secreções). M1 quando o escore pró supera o anti; M2 no caso contrário;
M0 quando estão equilibrados. Apenas M1 elimina o patógeno; M2 prioriza regulação
e reparo.

### Ações

| Ação | Efeito |
|---|---|
| `NOOP` | Não secreta nada |
| `SECRETE_PRO` | Secreta citocina pró-inflamatória |
| `SECRETE_ANTI` | Secreta citocina anti-inflamatória |
| `SECRETE_CHEMO` | Secreta quimiocina (recrutamento) |
| `SECRETE_REPAIR` | Secreta fator de reparo |

As ações antigas `ACTIVATE` e `REGULATE` continuam como aliases de
`SECRETE_PRO` e `SECRETE_ANTI`.

### Recompensa de equipe

```
R_t = (P_{t-1} - P_t)
      - lambda_I * Pro_t
      - lambda_D * D_t
      + lambda_H * max(0, D_{t-1} - D_t)
```

O episódio só termina quando o patógeno foi eliminado **e** o dano tecidual foi
recuperado (ou quando `max_steps` é atingido). Assim, o agente precisa aprender
a alternar entre resposta inflamatória e reparo.

## Instalação

Requer Python 3.13 e [uv](https://docs.astral.sh/uv/).

```bash
cd immune-rl
uv sync
```

## Uso

Demonstração rápida com narração no terminal:

```bash
uv run immune-rl
```

Treinar e visualizar o mundo emergente:

```bash
uv run python src/experimentos/visualize_graphs.py --episodes 600
uv run python src/experimentos/visualize_graphs.py --times 0 4 8 12 -1
uv run python src/experimentos/visualize_graphs.py --gif src/figures/evolucao.gif
uv run python src/experimentos/visualize_graphs.py --viewer
uv run python src/experimentos/visualize_graphs.py --narrate
```

Opções úteis:

- `--mode emergent|pilot`: mundo emergente (padrão) ou piloto reduzido.
- `--policy trained|random|pro|anti|chemo|repair`: política usada no episódio.
- `--episodes N`: episódios de treino (padrão 600).
- `--max-steps N`: passos por episódio (padrão 60).
- `--layout layered|spring|circular|shell`: layout do grafo.
- `--snapshots caminho.png`: figura com vários tempos.
- `--gif caminho.gif`: animação com linha do tempo e eventos.
- `--viewer`: janela interativa com slider, play/pause e teclado (`←`, `→`, espaço).

A figura animada mostra, em cada passo:

- o grafo com nós coloridos por tipo/fenótipo e arestas por relação;
- destaque amarelo nos nós/arestas que participaram dos eventos do passo;
- manchete e lista de eventos (`PAMP reconhecido`, `mac_2 recrutado`, etc.);
- linha do tempo com patógeno, citocinas, dano e contagem de M1/M2.

## Estrutura

```
src/immune_rl/
  entities.py       entidades, tipos de nó/aresta, ações e fenótipos
  events.py         eventos legíveis por passo
  graph.py          ImmuneGraph sobre NetworkX
  dynamics.py       dinâmicas e recompensas
  emergent.py       regras do mundo emergente
  environment.py    reset/step, histórico e narração
  encoding.py       discretização para a Q-table
  policies.py       políticas aleatória, fixa e Q-learning
  visualization.py  snapshots, animação e viewer interativo
src/experimentos/
  visualize_graphs.py  CLI de treino + visualização
tests/               testes unitários e de renderização
```

## Testes e lint

```bash
uv run pytest
uv run ruff check .
```

Os testes cobrem reconhecimento de PAMP, feedback pró-inflamatório, inibição por
IL-10, recrutamento, morte apenas por M1, dano/reparo, recompensa, alinhamento
dos eventos e geração de figuras/GIF em modo headless.
