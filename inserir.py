"""Carga do MovieLens no Neo4j.

Nós:     (:User {userId}), (:Movie {movieId, title, year}), (:Genre {name})
Arestas: (:User)-[:RATED {rating, timestamp, split}]->(:Movie)
         (:Movie)-[:IN_GENRE]->(:Genre)

As avaliações são divididas aqui em treino (80%) e teste (20%), dentro de cada
usuário. A divisão fica gravada na aresta RATED (atributo split), para que a
similaridade de Pearson seja calculada só com o treino.
"""
import random
from collections import defaultdict

from neo4j import GraphDatabase

from config import AUTH, URI
from dados import ler_avaliacoes, ler_filmes

PROPORCAO_TREINO = 0.8
SEMENTE = 42
TAMANHO_LOTE = 5000


def dividir_treino_teste(avaliacoes, proporcao=PROPORCAO_TREINO, semente=SEMENTE):
    """Marca cada avaliação como 'train' ou 'test', separando 80/20 por usuário."""
    gerador = random.Random(semente)
    por_usuario = defaultdict(list)
    for avaliacao in avaliacoes:
        por_usuario[avaliacao["userId"]].append(avaliacao)

    for user_id in sorted(por_usuario):
        lista = por_usuario[user_id]
        gerador.shuffle(lista)
        corte = int(len(lista) * proporcao)
        for posicao, avaliacao in enumerate(lista):
            avaliacao["split"] = "train" if posicao < corte else "test"
    return avaliacoes


def em_lotes(itens, tamanho=TAMANHO_LOTE):
    for inicio in range(0, len(itens), tamanho):
        yield itens[inicio:inicio + tamanho]


def limpar_banco(driver):
    """Apaga tudo, para o script poder ser rodado de novo do zero."""
    with driver.session() as sessao:
        sessao.run(
            "MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
        ).consume()


def criar_restricoes(driver):
    """Chaves únicas. Também criam índices, o que deixa os MATCH por id rápidos."""
    for comando in (
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE",
        "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE",
        "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
    ):
        driver.execute_query(comando)


def inserir_generos(driver, filmes):
    generos = sorted({g for f in filmes for g in f["genres"]})
    driver.execute_query(
        "UNWIND $generos AS nome MERGE (:Genre {name: nome})",
        generos=generos,
    )
    print(f"  {len(generos)} gêneros")


def inserir_filmes(driver, filmes):
    consulta = """
    UNWIND $lote AS f
    MERGE (m:Movie {movieId: f.movieId})
    SET m.title = f.title, m.year = f.year
    WITH m, f
    UNWIND f.genres AS nome_genero
    MATCH (g:Genre {name: nome_genero})
    MERGE (m)-[:IN_GENRE]->(g)
    """
    for lote in em_lotes(filmes, 1000):
        driver.execute_query(consulta, lote=lote)
    print(f"  {len(filmes)} filmes")


def inserir_usuarios(driver, avaliacoes):
    ids = sorted({a["userId"] for a in avaliacoes})
    driver.execute_query(
        "UNWIND $ids AS id MERGE (:User {userId: id})",
        ids=ids,
    )
    print(f"  {len(ids)} usuários")


def inserir_avaliacoes(driver, avaliacoes):
    consulta = """
    UNWIND $lote AS a
    MATCH (u:User {userId: a.userId})
    MATCH (m:Movie {movieId: a.movieId})
    MERGE (u)-[r:RATED]->(m)
    SET r.rating = a.rating, r.timestamp = a.timestamp, r.split = a.split
    """
    total = 0
    for lote in em_lotes(avaliacoes):
        driver.execute_query(consulta, lote=lote)
        total += len(lote)
        print(f"  {total}/{len(avaliacoes)} avaliações", end="\r")
    print()


def mostrar_resumo(driver):
    print("\nResumo do grafo:")
    for consulta in (
        "MATCH (n) RETURN labels(n)[0] AS tipo, count(*) AS total ORDER BY tipo",
        "MATCH ()-[r]->() RETURN type(r) AS tipo, count(*) AS total ORDER BY tipo",
        "MATCH ()-[r:RATED]->() RETURN 'RATED ' + r.split AS tipo, count(*) AS total ORDER BY tipo",
    ):
        for registro in driver.execute_query(consulta).records:
            print(f"  {registro['tipo']:<14} {registro['total']}")


def main():
    filmes = ler_filmes()
    avaliacoes = dividir_treino_teste(ler_avaliacoes())

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("Limpando o banco...")
        limpar_banco(driver)
        criar_restricoes(driver)

        print("Inserindo:")
        inserir_generos(driver, filmes)
        inserir_filmes(driver, filmes)
        inserir_usuarios(driver, avaliacoes)
        inserir_avaliacoes(driver, avaliacoes)
        mostrar_resumo(driver)


if __name__ == "__main__":
    main()