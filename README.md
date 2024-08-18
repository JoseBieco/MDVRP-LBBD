# MDVRP-LBBD

## Instâncias

Com o objetivo de observar o comportamento do modelo implementado, foram selecionadas as 5 primeiras instâncias do cunjunto presente em [neo.lcc][1].

### Estrutura do Arquivo

A primeira linha dos arquivos contém:

- **type m n t**

- **type**:
  - `0` (VRP)
  - `1` (PVRP)
  - `2` (MDVRP)
  - `3` (SDVRP)
  - `4` (VRPTW)
  - `5` (PVRPTW)
  - `6` (MDVRPTW)
  - `7` (SDVRPTW)
- **m**: número de veículos
- **n**: número de clientes
- **t**: número de dias (PVRP), depósitos (MDVRP) ou tipos de veículos (SDVRP)

#### Informações Adicionais

As próximas `t` linhas contêm, para cada dia (ou depósito ou tipo de veículo), as seguintes informações:

- **D Q**

- **D**: duração máxima de uma rota
- **Q**: carga máxima de um veículo

#### Informações dos Clientes

As linhas seguintes contêm, para cada cliente, as seguintes informações:

- **i x y d q f a list e l**

- **i**: número do cliente
- **x**: coordenada x
- **y**: coordenada y
- **d**: duração do serviço
- **q**: demanda
- **f**: frequência de visita
- **a**: número de combinações de visita possíveis
- **list**: lista de todas as combinações de visita possíveis
- **e**: início da janela de tempo (tempo mais cedo para começar o serviço), se houver
- **l**: fim da janela de tempo (tempo mais tarde para começar o serviço), se houver

### Definição

As instâncias foram classificadas em três categorias: `Pequeno`, `Médio` e `Grande`, com os seguintes parâmetros:

- **Pequeno**
  - Veículos: 4
  - Clientes: 15
  - Depósitos: 2
  - Capacidade: 80

- **Médio**
  - Veículos: 4
  - Clientes: 25
  - Depósitos: 4
  - Capacidade: 80

- **Grande**
  - Veículos: 4
  - Clientes: 50
  - Depósitos: 4
  - Capacidade: 80

Os parâmetros foram estabelecidos após uma análise das instâncias, levando em consideração que o **MDVRP** é um problema NP-Difícil. Isso implica que o tempo necessário para resolver o problema pode crescer exponencialmente com o aumento das variáveis.

> O problema NP-Difícil refere-se à dificuldade de encontrar uma solução ótima em tempo polinomial, o que significa que é computacionalmente intensivo para grandes instâncias.

Foi realizado um cuidado especial para garantir que a `capacidade` e a `quantidade de veículos` fossem suficientes para atender a todas as demandas do problema. Especificamente, a soma das demandas \( D \) de todos os clientes \( C \) deve ser menor ou igual ao produto da quantidade de veículos \( V \), a quantidade de depósitos \( d \) e a capacidade de cada veículo (80).

Matematicamente, isso pode ser representado como:

$$
    \sum_{i=1}^{C} d_i \leq V \times D \times 80
$$

## Referências

[1]: https://neo.lcc.uma.es/vrp/vrp-instances/multiple-depot-vrp-instances/
