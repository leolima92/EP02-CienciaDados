# EP02 – Sistema de Recomendação de Filmes com Grafos

Sistema de recomendação baseado em **filtragem colaborativa** (similaridade entre filmes + regressão com KNN) sobre o dataset **MovieLens**, modelado como um grafo no **Neo4j**.

**Grupo:**
- Eduardo Leal Aruth
- Leonardo Rodrigues de Lima
- Yasmin Ayumi Foltran Mano

---

## Sumário

1. [Modelagem do grafo](#1-modelagem-do-grafo)
2. [Sistema de recomendação](#2-sistema-de-recomendação)
3. [Avaliação do modelo](#3-avaliação-do-modelo)
4. [Estrutura do projeto](#estrutura-do-projeto)
5. [Como executar](#como-executar)
6. [Consultas de exemplo](#consultas-de-exemplo)

---

## 1. Modelagem do grafo

Usamos apenas `movies.csv` e `ratings.csv` do **MovieLens Latest Small** (100.836 avaliações, 9.742 filmes, 610 usuários).

<img width="1387" height="590" alt="image" src="https://github.com/user-attachments/assets/34b36e31-b527-45b9-b2cd-c1c71b800b03" />


### Nós

| Rótulo  | Atributos                                    | Origem                                                          |
|---------|----------------------------------------------|-----------------------------------------------------------------|
| `User`  | `userId`                                     | `ratings.csv`                                                   |
| `Movie` | `movieId`, `title`, `year`, `media_treino`, `qtd_treino` | `movies.csv` (`year` é extraído do título; as médias são calculadas) |
| `Genre` | `name`                                       | coluna `genres` do `movies.csv`, separada por `\|`              |

### Arestas

| Tipo                            | Atributos                         | Significado                                         |
|---------------------------------|-----------------------------------|-----------------------------------------------------|
| `(:User)-[:RATED]->(:Movie)`    | `rating`, `timestamp`, `split`    | O usuário avaliou o filme                           |
| `(:Movie)-[:IN_GENRE]->(:Genre)`| –                                 | O filme pertence ao gênero                          |
| `(:Movie)-[:SIMILAR]->(:Movie)` | `pearson`, `comuns`               | Correlação de Pearson entre os dois filmes          |

### Decisões de modelagem

- **Gênero como nó, e não como lista dentro do filme:** as consultas por gênero viram caminhos no grafo (`Movie → Genre`), e cada gênero existe uma vez só.
- **A nota fica na aresta `RATED`:** ela não pertence ao usuário nem ao filme, e sim à relação entre os dois.
- **Divisão treino/teste na própria aresta (`split = "train" | "test"`):** a divisão 80/20 é feita na carga, por usuário e com semente fixa (`42`). Assim o cálculo de similaridade usa só o treino, e o resultado é reprodutível.
- **Filmes com `(no genres listed)`** ficam sem aresta `IN_GENRE`, em vez de criar um gênero falso.
- **`SIMILAR` é gravada uma vez por par**, já que `c(i,j) = c(j,i)`, e é consultada sem direção: `(a)-[:SIMILAR]-(b)`.

---

## 2. Sistema de recomendação

### Correlação de Pearson entre filmes

Como cada usuário avalia poucos filmes, a similaridade é calculada **entre filmes**:

$$
c(i,j) = \frac{\sum_{k} (r_i^k - \bar{r}_i)(r_j^k - \bar{r}_j)}{\sqrt{\sum_{k} (r_i^k - \bar{r}_i)^2} \sqrt{\sum_{k} (r_j^k - \bar{r}_j)^2}}
$$

- $r_i^k$: nota que o usuário $k$ deu ao filme $i$
- $\bar{r}_i$: nota média do filme $i$ (no conjunto de treino)
- $k$ percorre os usuários que avaliaram os dois filmes

Critérios para gravar a aresta `SIMILAR`:

- **Pelo menos 10 usuários em comum.** Com poucos usuários, a correlação é dominada por coincidências.
- **Somente correlações positivas.** O KNN usa os vizinhos mais parecidos.
- **Somente avaliações de treino** entram no cálculo.

### Regressão com KNN

A nota que o usuário $u$ provavelmente daria ao filme $i$ é a média das notas que ele deu aos $k$ filmes mais parecidos com $i$, ponderada pela correlação:

$$
\hat{r}^u_i = \frac{\sum_{j \in \mathcal{N}_k(i;u)} c(i,j) \cdot r^u_j}{\sum_{j \in \mathcal{N}_k(i;u)} |c(i,j)|}
$$

onde $\mathcal{N}_k(i;u)$ são os $k$ filmes com maior $c(i,j)$ entre os que o usuário $u$ avaliou.

> **Observação sobre o enunciado:** na fórmula do enunciado, o numerador usa $r_i^k$ e soma em $\mathcal{N}$, enquanto o denominador soma em $\mathcal{N}_k(x)$. Interpretamos como acima: a nota usada é a do **usuário $u$ no filme vizinho $j$** ($r^u_j$), e os dois somatórios percorrem o **mesmo conjunto** de vizinhos.

*Em desenvolvimento.*

---

## 3. Avaliação do modelo

- Divisão **80% treino / 20% teste**, por usuário.
- Métricas: **precisão** e **recall** sobre as avaliações de teste.

*Em desenvolvimento.*

---

## Estrutura do projeto

```
.
├── docker-compose.yml   # Neo4j 5.26 local
├── requirements.txt     # apenas o driver neo4j
├── config.py            # URI e credenciais do Neo4j
├── dados.py             # leitura dos CSVs (biblioteca padrão)
├── testar_conexao.py    # teste rápido de conexão
├── inserir.py           # carga: User, Movie, Genre, RATED, IN_GENRE + divisão 80/20
├── similaridade.py      # Pearson entre filmes -> arestas SIMILAR
├── docs/
│   └── modelo-grafo.png # print do schema (CALL db.schema.visualization())
└── data/                # CSVs do MovieLens (não versionados)
```

Conforme o enunciado, não usamos bibliotecas além do que o problema pede: só o driver `neo4j` e a biblioteca padrão do Python (`csv`, `random`, `math`...).

---

## Como executar

### Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Python 3.10+

### 1. Subir o Neo4j

```bash
docker compose up -d
```

Aguarde cerca de 30 segundos, até `http://localhost:7474` abrir.

- **Usuário:** `neo4j`
- **Senha:** `movielens123`

### 2. Ambiente Python

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Dataset

A licença do MovieLens não permite redistribuir os dados, por isso eles não estão no repositório.

1. Baixe o [ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip).
2. Extraia `movies.csv` e `ratings.csv` para a pasta `data/`.

### 4. Executar, nesta ordem

```bash
python testar_conexao.py   # deve imprimir "conectou"
python inserir.py          # carga do grafo + divisão treino/teste
python similaridade.py     # correlação de Pearson -> arestas SIMILAR
```

Para apagar tudo e recomeçar do zero: `docker compose down -v`.

---

## Consultas de exemplo

**Quantos filmes de comédia o usuário 1 avaliou com nota 4 ou mais?**

```cypher
MATCH (u:User {userId: 1})-[r:RATED]->(m:Movie)-[:IN_GENRE]->(:Genre {name: 'Comedy'})
WHERE r.rating >= 4
RETURN count(m) AS comedias_nota_4_ou_mais
```

Resultado: **70**

**Filmes mais parecidos com Toy Story:**

```cypher
MATCH (:Movie {movieId: 1})-[s:SIMILAR]-(outro:Movie)
RETURN outro.title AS filme, round(s.pearson, 3) AS pearson, s.comuns AS usuarios_em_comum
ORDER BY s.pearson DESC
LIMIT 10
```

**Visualizar o modelo do grafo:**

```cypher
CALL db.schema.visualization()
```
