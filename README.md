# EP02 – Sistema de Recomendação de Filmes com Grafos

Sistema de recomendação baseado em **filtragem colaborativa** (similaridade entre filmes + regressão com KNN) sobre o dataset **MovieLens**, modelado como um grafo no **Neo4j**.

**Grupo:**
- Eduardo Leal Aruth
- Leonardo Rodrigues de Lima
- Yasmin Ayumi Foltran Mano

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

- **Pelo menos 10 usuários em comum.** Com poucos usuários, a correlação é dominada por coincidências. O valor pode ser trocado na linha de comando: `python similaridade.py 20`.
- **Somente correlações positivas** (acima de `1e-9`, para descartar zeros que viram positivos por arredondamento de ponto flutuante). O KNN usa os vizinhos mais parecidos.
- **Somente avaliações de treino** entram no cálculo.

**Resultado com os parâmetros padrão:**

| Item                                              | Valor   |
|---------------------------------------------------|---------|
| Avaliações de treino                              | 80.419  |
| Filmes com alguma avaliação de treino             | 8.933   |
| Filmes com pelo menos 10 avaliações de treino     | 1.931   |
| Arestas `SIMILAR` criadas                         | 148.675 |
| Maior correlação                                  | 0,962   |

Só os 1.931 filmes com pelo menos 10 avaliações de treino podem ter vizinhos. Para os demais, o KNN não consegue prever nota, e é por isso que a avaliação também mede a **cobertura**.

**Verificação do cálculo:** recalculamos a correlação de um par direto em Cypher, sem passar pelo Python, e comparamos com o valor gravado na aresta. Para Toy Story × Toy Story 2, os dois deram **0,63924**, com os mesmos **55** usuários em comum.

```cypher
MATCH (a:Movie {movieId: 1}), (b:Movie {movieId: 3114})
MATCH (a)<-[ra:RATED {split: 'train'}]-(:User)-[rb:RATED {split: 'train'}]->(b)
WITH a, b, collect({x: ra.rating - a.media_treino, y: rb.rating - b.media_treino}) AS d
WITH a, b, size(d) AS comuns,
     reduce(s = 0.0, p IN d | s + p.x * p.y) AS num,
     reduce(s = 0.0, p IN d | s + p.x ^ 2) AS sx,
     reduce(s = 0.0, p IN d | s + p.y ^ 2) AS sy
OPTIONAL MATCH (a)-[s:SIMILAR]-(b)
RETURN comuns, num / sqrt(sx * sy) AS pearson_cypher, s.pearson AS pearson_gravado, s.comuns AS comuns_gravado
```

### Regressão com KNN

A nota que o usuário $u$ provavelmente daria ao filme $i$ é a média das notas que ele deu aos $k$ filmes mais parecidos com $i$, ponderada pela correlação:

$$
\hat{r}^u_i = \frac{\sum_{j \in \mathcal{N}_k(i;u)} c(i,j) \cdot r^u_j}{\sum_{j \in \mathcal{N}_k(i;u)} |c(i,j)|}
$$

onde $\mathcal{N}_k(i;u)$ são os $k$ filmes com maior $c(i,j)$ entre os que o usuário $u$ avaliou.

> **Observação sobre o enunciado:** na fórmula do enunciado, o numerador usa $r_i^k$ e soma em $\mathcal{N}$, enquanto o denominador soma em $\mathcal{N}_k(x)$. Interpretamos como acima: a nota usada é a do **usuário $u$ no filme vizinho $j$** ($r^u_j$), e os dois somatórios percorrem o **mesmo conjunto** de vizinhos.

A previsão é calculada direto no Neo4j, percorrendo `User -RATED-> Movie -SIMILAR- Movie`: a partir dos filmes que o usuário avaliou no treino, o grafo chega aos filmes parecidos.

**Parâmetros:**

- **`k = 30` vizinhos** na média ponderada.
- **Mínimo de 3 vizinhos** para um filme entrar nas recomendações. Com um vizinho só, a previsão é uma cópia da nota que o usuário deu a ele.
- **Candidatos a recomendação:** os filmes que o usuário **não avaliou no treino**.

Os três são configuráveis pela linha de comando.

**Uso:**

```bash
python recomendar.py 1                  # top 10 recomendações para o usuário 1
python recomendar.py 1 --filme 2571     # nota prevista do usuário 1 para Matrix
python recomendar.py 1 --k 20 --n 15 --min-vizinhos 10
```

**Exemplo:** previsões para filmes que o usuário 1 tem no conjunto de teste (o modelo não viu essas notas).

| Filme             | Nota real | Nota prevista | Vizinhos |
|-------------------|-----------|---------------|----------|
| Schindler's List  | 5,0       | 4,64          | 30       |
| Matrix            | 5,0       | 4,34          | 30       |
| Starship Troopers | 3,0       | 4,13          | 30       |
| Blown Away        | 3,0       | sem previsão  | 0        |

O modelo acerta a **ordem**: o filme de que o usuário gostou menos recebe a menor previsão. Mas ele comprime a **escala**. Como a fórmula é uma média ponderada das notas do próprio usuário, quem dá notas altas para quase tudo recebe previsões altas para quase tudo. Filmes sem nenhum vizinho avaliado pelo usuário ficam sem previsão.

**Efeito do mínimo de vizinhos nas recomendações:** com o mínimo de 3, o topo da lista do usuário 1 fica cheio de previsões 5,00 baseadas em só 3 vizinhos. Com `--min-vizinhos 10`, entram filmes com mais evidência, como *American Gangster*, com 30 vizinhos e previsão de 4,76. O valor padrão vai ser escolhido com base na avaliação.

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
├── recomendar.py        # KNN: nota prevista e top recomendações por usuário
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
python recomendar.py 1     # top 10 recomendações para o usuário 1
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

No topo aparecem filmes como *Dirty Harry* e *The Apartment*, com Pearson acima de 0,75 mas só de 10 a 12 usuários em comum, enquanto *Aladdin* tem 0,71 com 66 usuários em comum. Correlações com poucos usuários são menos confiáveis, e por isso o mínimo de usuários em comum também entra no experimento da avaliação.

**Top recomendações para o usuário X** (o mesmo cálculo do `recomendar.py`, com X = 1):

```cypher
MATCH (u:User {userId: 1})-[r:RATED {split: 'train'}]->(:Movie)-[s:SIMILAR]-(i:Movie)
WHERE NOT EXISTS { (u)-[:RATED {split: 'train'}]->(i) }
WITH u, i, s.pearson AS c, r.rating AS nota
ORDER BY c DESC
WITH u, i, collect({c: c, nota: nota})[..30] AS vizinhos
WHERE size(vizinhos) >= 3
RETURN i.title AS filme,
       reduce(n = 0.0, v IN vizinhos | n + v.c * v.nota)
     / reduce(d = 0.0, v IN vizinhos | d + abs(v.c)) AS nota_prevista,
       size(vizinhos) AS vizinhos
ORDER BY nota_prevista DESC, vizinhos DESC
LIMIT 10
```

**Visualizar o modelo do grafo:**

```cypher
CALL db.schema.visualization()
```