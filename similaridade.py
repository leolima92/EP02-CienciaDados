"""Similaridade de Pearson entre filmes, gravada no grafo.

    c(i,j) = Σk (r_i^k - r̄_i)(r_j^k - r̄_j)
             / ( sqrt(Σk (r_i^k - r̄_i)²) · sqrt(Σk (r_j^k - r̄_j)²) )

- k percorre os usuários que avaliaram os dois filmes (só no treino)
- r̄_i é a nota média do filme i no treino

Resultado no grafo: (:Movie)-[:SIMILAR {pearson, comuns}]->(:Movie)
"""
import math
import sys
import time
from collections import defaultdict

from neo4j import GraphDatabase

from config import AUTH, URI

MIN_USUARIOS_EM_COMUM = 10   # pode ser trocado na linha de comando: python similaridade.py 20
SOMENTE_POSITIVAS = True
PEARSON_MINIMO = 1e-9         # abaixo disso é zero na prática (arredondamento de ponto flutuante)
TAMANHO_LOTE = 10000

CONSULTA_SIMILAR = """
UNWIND $lote AS p
MATCH (a:Movie {movieId: p.i})
MATCH (b:Movie {movieId: p.j})
CREATE (a)-[:SIMILAR {pearson: p.pearson, comuns: p.comuns}]->(b)
"""


def ler_treino(driver):
    registros = driver.execute_query(
        """
        MATCH (u:User)-[r:RATED {split: 'train'}]->(m:Movie)
        RETURN u.userId AS usuario, m.movieId AS filme, r.rating AS nota
        """
    ).records
    return [(r["usuario"], r["filme"], r["nota"]) for r in registros]


def calcular_medias(avaliacoes):
    soma = defaultdict(float)
    quantidade = defaultdict(int)
    for _, filme, nota in avaliacoes:
        soma[filme] += nota
        quantidade[filme] += 1
    medias = {filme: soma[filme] / quantidade[filme] for filme in soma}
    return medias, quantidade


def gravar_medias(driver, medias, quantidades):
    linhas = [
        {"filme": f, "media": m, "qtd": quantidades[f]} for f, m in medias.items()
    ]
    for inicio in range(0, len(linhas), TAMANHO_LOTE):
        driver.execute_query(
            """
            UNWIND $lote AS x
            MATCH (m:Movie {movieId: x.filme})
            SET m.media_treino = x.media, m.qtd_treino = x.qtd
            """,
            lote=linhas[inicio:inicio + TAMANHO_LOTE],
        )


def apagar_similaridades(driver):
    """Permite rodar o script de novo sem duplicar arestas."""
    with driver.session() as sessao:
        sessao.run(
            "MATCH ()-[s:SIMILAR]->() CALL (s) { DELETE s } IN TRANSACTIONS OF 10000 ROWS"
        ).consume()


def gerar_pares(avaliacoes, medias, quantidades, minimo=MIN_USUARIOS_EM_COMUM):
    """Gera os pares (i, j), com i < j, e a correlação de Pearson entre eles."""
    # Filme com menos de `minimo` avaliações nunca terá `minimo` usuários em comum.
    elegiveis = {f for f, n in quantidades.items() if n >= minimo}

    # Para cada usuário: lista (filme, nota - média do filme), ordenada por filme.
    por_usuario = defaultdict(list)
    for usuario, filme, nota in avaliacoes:
        if filme in elegiveis:
            por_usuario[usuario].append((filme, nota - medias[filme]))

    # Para cada filme: quais usuários o avaliaram e em que posição da lista deles.
    usuarios_do_filme = defaultdict(list)
    for usuario, lista in por_usuario.items():
        lista.sort()
        for posicao, (filme, _) in enumerate(lista):
            usuarios_do_filme[filme].append((usuario, posicao))

    filmes = sorted(elegiveis)
    inicio = time.time()
    for contador, i in enumerate(filmes, 1):
        # j -> [numerador, Σ desvio_i², Σ desvio_j², usuários em comum]
        acumulado = {}
        for usuario, posicao in usuarios_do_filme[i]:
            lista = por_usuario[usuario]
            desvio_i = lista[posicao][1]
            for j, desvio_j in lista[posicao + 1:]:   # só j > i
                a = acumulado.get(j)
                if a is None:
                    acumulado[j] = [desvio_i * desvio_j, desvio_i ** 2, desvio_j ** 2, 1]
                else:
                    a[0] += desvio_i * desvio_j
                    a[1] += desvio_i ** 2
                    a[2] += desvio_j ** 2
                    a[3] += 1

        for j, (numerador, soma_i, soma_j, comuns) in acumulado.items():
            if comuns < minimo or soma_i < 1e-12 or soma_j < 1e-12:
                continue
            c = numerador / (math.sqrt(soma_i) * math.sqrt(soma_j))
            if SOMENTE_POSITIVAS and c <= PEARSON_MINIMO:
                continue
            yield {"i": i, "j": j, "pearson": c, "comuns": comuns}

        if contador % 250 == 0 or contador == len(filmes):
            print(f"  {contador}/{len(filmes)} filmes ({time.time() - inicio:.0f}s)", end="\r")
    print()


def main():
    minimo = int(sys.argv[1]) if len(sys.argv) > 1 else MIN_USUARIOS_EM_COMUM

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()

        print("Lendo avaliações de treino do grafo...")
        avaliacoes = ler_treino(driver)
        medias, quantidades = calcular_medias(avaliacoes)
        print(f"  {len(avaliacoes)} avaliações de treino, {len(medias)} filmes")

        print("Gravando a nota média de treino em cada filme...")
        gravar_medias(driver, medias, quantidades)

        print("Apagando similaridades antigas...")
        apagar_similaridades(driver)

        print(f"Calculando Pearson (mínimo de {minimo} usuários em comum)...")
        lote, total = [], 0
        for par in gerar_pares(avaliacoes, medias, quantidades, minimo):
            lote.append(par)
            if len(lote) >= TAMANHO_LOTE:
                driver.execute_query(CONSULTA_SIMILAR, lote=lote)
                total += len(lote)
                lote = []
        if lote:
            driver.execute_query(CONSULTA_SIMILAR, lote=lote)
            total += len(lote)
        print(f"{total} arestas SIMILAR criadas")


if __name__ == "__main__":
    main()