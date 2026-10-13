"""Regressão com KNN sobre o grafo: previsão de nota e top recomendações.

Nota prevista do usuário u para o filme i:

    r̂_i^u = Σ_{j ∈ N_k(i;u)} c(i,j) · r_j^u  /  Σ_{j ∈ N_k(i;u)} |c(i,j)|

N_k(i;u) = os k filmes mais parecidos com i (maior Pearson) entre os que u avaliou
no treino. Tudo é calculado no Neo4j, percorrendo User -RATED-> Movie -SIMILAR- Movie.

Uso:
    python recomendar.py 1                 # top 10 recomendações para o usuário 1
    python recomendar.py 1 --filme 3114    # nota prevista do usuário 1 para Toy Story 2
    python recomendar.py 1 --k 20 --n 15   # muda o k do KNN e o tamanho da lista
"""
import argparse

from neo4j import GraphDatabase

from config import AUTH, URI

K_PADRAO = 30            # vizinhos usados na média ponderada
MIN_VIZINHOS_PADRAO = 3  # abaixo disso a previsão é instável (1 vizinho = copiar a nota dele)

CONSULTA_PREVER_NOTA = """
MATCH (u:User {userId: $usuario})-[r:RATED {split: 'train'}]->(:Movie)-[s:SIMILAR]-(i:Movie {movieId: $filme})
WITH s.pearson AS c, r.rating AS nota
ORDER BY c DESC
LIMIT $k
WITH collect({c: c, nota: nota}) AS vizinhos
RETURN size(vizinhos) AS qtd_vizinhos,
       CASE WHEN size(vizinhos) = 0 THEN null
            ELSE reduce(n = 0.0, v IN vizinhos | n + v.c * v.nota)
               / reduce(d = 0.0, v IN vizinhos | d + abs(v.c))
       END AS nota_prevista
"""

CONSULTA_TOP_RECOMENDACOES = """
MATCH (u:User {userId: $usuario})-[r:RATED {split: 'train'}]->(:Movie)-[s:SIMILAR]-(i:Movie)
WHERE NOT EXISTS { (u)-[:RATED {split: 'train'}]->(i) }
WITH u, i, s.pearson AS c, r.rating AS nota
ORDER BY c DESC
WITH u, i, collect({c: c, nota: nota})[..$k] AS vizinhos
WHERE size(vizinhos) >= $min_vizinhos
WITH u, i, size(vizinhos) AS qtd_vizinhos,
     reduce(n = 0.0, v IN vizinhos | n + v.c * v.nota)
   / reduce(d = 0.0, v IN vizinhos | d + abs(v.c)) AS nota_prevista
ORDER BY nota_prevista DESC, qtd_vizinhos DESC
LIMIT $n
OPTIONAL MATCH (u)-[t:RATED {split: 'test'}]->(i)
RETURN i.movieId AS movieId, i.title AS titulo, nota_prevista, qtd_vizinhos,
       t.rating AS nota_real_teste
ORDER BY nota_prevista DESC, qtd_vizinhos DESC
"""

CONSULTA_FAVORITOS = """
MATCH (:User {userId: $usuario})-[r:RATED {split: 'train'}]->(m:Movie)
RETURN m.title AS titulo, r.rating AS nota
ORDER BY r.rating DESC, m.title
LIMIT $n
"""


def prever_nota(driver, usuario, filme, k=K_PADRAO):
    """Devolve (nota_prevista, qtd_vizinhos). A nota vem None se não houver vizinhos."""
    registro = driver.execute_query(
        CONSULTA_PREVER_NOTA, usuario=usuario, filme=filme, k=k
    ).records[0]
    return registro["nota_prevista"], registro["qtd_vizinhos"]


def top_recomendacoes(driver, usuario, n=10, k=K_PADRAO, min_vizinhos=MIN_VIZINHOS_PADRAO):
    """Filmes que o usuário não avaliou no treino, ordenados pela nota prevista."""
    return driver.execute_query(
        CONSULTA_TOP_RECOMENDACOES,
        usuario=usuario, n=n, k=k, min_vizinhos=min_vizinhos,
    ).records


def favoritos(driver, usuario, n=5):
    return driver.execute_query(CONSULTA_FAVORITOS, usuario=usuario, n=n).records


def formatar_nota(nota):
    return "  -  " if nota is None else f"{nota:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Recomendação de filmes com KNN no Neo4j")
    parser.add_argument("usuario", type=int, help="userId do MovieLens")
    parser.add_argument("--filme", type=int, help="movieId para prever uma nota específica")
    parser.add_argument("--k", type=int, default=K_PADRAO, help="vizinhos do KNN")
    parser.add_argument("--n", type=int, default=10, help="tamanho da lista de recomendações")
    parser.add_argument("--min-vizinhos", type=int, default=MIN_VIZINHOS_PADRAO)
    args = parser.parse_args()

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()

        if args.filme is not None:
            nota, qtd = prever_nota(driver, args.usuario, args.filme, args.k)
            if nota is None:
                print(f"Sem vizinhos: o usuário {args.usuario} não avaliou nenhum filme parecido com {args.filme}.")
            else:
                print(f"Nota prevista do usuário {args.usuario} para o filme {args.filme}: "
                      f"{nota:.2f} (com {qtd} vizinhos)")
            return

        print(f"Filmes que o usuário {args.usuario} mais gostou (treino):")
        for registro in favoritos(driver, args.usuario):
            print(f"  {registro['nota']:.1f}  {registro['titulo']}")

        print(f"\nTop {args.n} recomendações (k={args.k}, mínimo de {args.min_vizinhos} vizinhos):")
        print(f"  {'prevista':>8}  {'real*':>5}  {'viz':>3}  filme")
        for registro in top_recomendacoes(
            driver, args.usuario, args.n, args.k, args.min_vizinhos
        ):
            print(f"  {registro['nota_prevista']:>8.2f}  "
                  f"{formatar_nota(registro['nota_real_teste']):>5}  "
                  f"{registro['qtd_vizinhos']:>3}  {registro['titulo']}")
        print("\n* real = nota que o usuário deu no conjunto de teste (o modelo não viu); '-' = não avaliou")


if __name__ == "__main__":
    main()