"""Avaliação do modelo (item 3 do enunciado).

Para cada avaliação de teste (usuário u, filme i, nota real), o modelo prevê a
nota usando só o treino, com o mesmo KNN do recomendar.py:

    r̂_i^u = Σ c(i,j) · r_j^u / Σ |c(i,j)|,   j ∈ k vizinhos de i avaliados por u

Os vizinhos de cada caso de teste são buscados no grafo uma vez só
(User -RATED-> Movie -SIMILAR- Movie). Depois a previsão é refeita em memória
para várias combinações de parâmetros, o que permite comparar configurações sem
recalcular o Pearson: filtrar as arestas com `comuns >= 20` dá exatamente o
mesmo resultado que rodar `python similaridade.py 20`.

Métricas:
  - Cobertura: % das avaliações de teste para as quais o modelo previu uma nota
  - Precisão / Recall com limiar: recomendado = previsão >= 4, relevante = nota real >= 4
      precisão = recomendados relevantes / recomendados
      recall   = recomendados relevantes / todos os relevantes do teste
                 (relevantes sem previsão contam como não recomendados)
  - Precision@10 / Recall@10: para cada usuário, os 10 filmes de teste com maior
    previsão; média entre os usuários que têm algum relevante no teste
  - MAE / RMSE: erro médio da nota prevista (extra)

Uso:
    python avaliar.py
"""
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase

from config import AUTH, URI

LIMIAR_RELEVANTE = 4.0
TOP_N = 10
PADRAO = {"k": 30, "min_comuns": 10, "min_vizinhos": 3}

GRADE_K = (10, 20, 30, 50)
GRADE_MIN_COMUNS = (10, 20, 30)
GRADE_MIN_VIZINHOS = (1, 3, 10)

ARQUIVO_RESULTADOS = Path(__file__).parent / "resultados_avaliacao.csv"
LOTE_USUARIOS = 50

CONSULTA_CASOS_TESTE = """
UNWIND $usuarios AS uid
MATCH (u:User {userId: uid})-[t:RATED {split: 'test'}]->(i:Movie)
OPTIONAL MATCH (u)-[r:RATED {split: 'train'}]->(:Movie)-[s:SIMILAR]-(i)
WITH uid, i, t.rating AS nota_real,
     collect(CASE WHEN s IS NULL THEN null
                  ELSE [s.pearson, s.comuns, r.rating] END) AS vizinhos
RETURN uid AS usuario, i.movieId AS filme, nota_real, vizinhos
"""


def buscar_casos(driver):
    """Lista de (usuario, filme, nota_real, vizinhos), com vizinhos = [(pearson, comuns, nota)]
    em ordem decrescente de Pearson."""
    usuarios = [
        r["id"]
        for r in driver.execute_query("MATCH (u:User) RETURN u.userId AS id ORDER BY id").records
    ]
    casos = []
    for inicio in range(0, len(usuarios), LOTE_USUARIOS):
        lote = usuarios[inicio:inicio + LOTE_USUARIOS]
        for r in driver.execute_query(CONSULTA_CASOS_TESTE, usuarios=lote).records:
            vizinhos = sorted((tuple(v) for v in r["vizinhos"]), key=lambda v: v[0], reverse=True)
            casos.append((r["usuario"], r["filme"], r["nota_real"], vizinhos))
        print(f"  {min(inicio + LOTE_USUARIOS, len(usuarios))}/{len(usuarios)} usuários", end="\r")
    print()
    return casos


def prever(vizinhos, k, min_comuns, min_vizinhos):
    """KNN: média das notas dos k vizinhos mais parecidos, ponderada pelo Pearson."""
    escolhidos = []
    for c, comuns, nota in vizinhos:
        if comuns >= min_comuns:
            escolhidos.append((c, nota))
            if len(escolhidos) == k:
                break
    if len(escolhidos) < max(1, min_vizinhos):
        return None, len(escolhidos)
    numerador = sum(c * nota for c, nota in escolhidos)
    denominador = sum(abs(c) for c, _ in escolhidos)
    return numerador / denominador, len(escolhidos)


def avaliar(casos, k, min_comuns, min_vizinhos, limiar=LIMIAR_RELEVANTE, top_n=TOP_N):
    previstos = []                       # (usuario, filme, real, prevista, qtd_vizinhos)
    relevantes_por_usuario = defaultdict(int)
    for usuario, filme, real, vizinhos in casos:
        if real >= limiar:
            relevantes_por_usuario[usuario] += 1
        prevista, qtd = prever(vizinhos, k, min_comuns, min_vizinhos)
        if prevista is not None:
            previstos.append((usuario, filme, real, prevista, qtd))

    relevantes_total = sum(relevantes_por_usuario.values())

    # Precisão e recall com limiar
    recomendados = [p for p in previstos if p[3] >= limiar]
    acertos = sum(1 for p in recomendados if p[2] >= limiar)
    precisao = acertos / len(recomendados) if recomendados else 0.0
    recall = acertos / relevantes_total if relevantes_total else 0.0

    # Erro da nota prevista
    erros = [p[3] - p[2] for p in previstos]
    mae = sum(abs(e) for e in erros) / len(erros) if erros else float("nan")
    rmse = math.sqrt(sum(e * e for e in erros) / len(erros)) if erros else float("nan")

    # Precision@N e Recall@N por usuário
    por_usuario = defaultdict(list)
    for p in previstos:
        por_usuario[p[0]].append(p)

    precisoes_n, recalls_n, precisoes_aleatorias = [], [], []
    for usuario, relevantes in relevantes_por_usuario.items():
        if relevantes == 0:
            continue
        itens = por_usuario.get(usuario, [])
        if not itens:                     # tem relevantes, mas nenhuma previsão
            recalls_n.append(0.0)
            continue
        itens.sort(key=lambda p: (-p[3], -p[4], p[1]))
        topo = itens[:top_n]
        acertos_topo = sum(1 for p in topo if p[2] >= limiar)
        precisoes_n.append(acertos_topo / len(topo))
        recalls_n.append(acertos_topo / relevantes)
        # referência: precisão esperada se os filmes fossem ordenados ao acaso
        precisoes_aleatorias.append(sum(1 for p in itens if p[2] >= limiar) / len(itens))

    def media(valores):
        return sum(valores) / len(valores) if valores else 0.0

    return {
        "k": k,
        "min_comuns": min_comuns,
        "min_vizinhos": min_vizinhos,
        "previstos": len(previstos),
        "cobertura": len(previstos) / len(casos),
        "precisao": precisao,
        "recall": recall,
        f"precisao@{top_n}": media(precisoes_n),
        f"recall@{top_n}": media(recalls_n),
        f"precisao@{top_n}_aleatoria": media(precisoes_aleatorias),
        "mae": mae,
        "rmse": rmse,
    }


def porcentagem(valor):
    return f"{100 * valor:5.1f}%"


def main():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("Buscando no grafo os vizinhos de cada avaliação de teste...")
        inicio = time.time()
        casos = buscar_casos(driver)
    print(f"  {len(casos)} avaliações de teste ({time.time() - inicio:.0f}s)")

    relevantes = sum(1 for c in casos if c[2] >= LIMIAR_RELEVANTE)
    print(f"  {porcentagem(relevantes / len(casos))} delas são relevantes (nota real >= {LIMIAR_RELEVANTE})"
          " -> é a precisão de quem recomendasse tudo")

    # Conferência com o recomendar.py (usuário 1, Matrix)
    for usuario, filme, real, vizinhos in casos:
        if usuario == 1 and filme == 2571:
            prevista, qtd = prever(vizinhos, PADRAO["k"], PADRAO["min_comuns"], 1)
            print(f"  Conferência: usuário 1, Matrix -> prevista {prevista:.2f} com {qtd} vizinhos"
                  f" (o recomendar.py deu 4.34 com 30)")

    n = TOP_N
    r = avaliar(casos, **PADRAO)
    print(f"\nConfiguração padrão: k={r['k']}, mínimo de {r['min_comuns']} usuários em comum,"
          f" mínimo de {r['min_vizinhos']} vizinhos")
    print(f"  Cobertura:            {porcentagem(r['cobertura'])}  ({r['previstos']} de {len(casos)})")
    print(f"  Precisão (limiar 4):  {porcentagem(r['precisao'])}")
    print(f"  Recall   (limiar 4):  {porcentagem(r['recall'])}")
    print(f"  Precision@{n}:        {porcentagem(r[f'precisao@{n}'])}"
          f"  (ordem aleatória: {porcentagem(r[f'precisao@{n}_aleatoria'])})")
    print(f"  Recall@{n}:           {porcentagem(r[f'recall@{n}'])}")
    print(f"  MAE / RMSE:           {r['mae']:.3f} / {r['rmse']:.3f}")

    print("\nComparação de parâmetros:")
    cabecalho = (f"  {'k':>3} {'min_com':>7} {'min_viz':>7} {'cobert':>7} {'prec':>7} {'recall':>7}"
                 f" {'P@' + str(n):>7} {'R@' + str(n):>7} {'MAE':>6} {'RMSE':>6}")
    print(cabecalho)
    resultados = []
    for min_comuns in GRADE_MIN_COMUNS:
        for min_vizinhos in GRADE_MIN_VIZINHOS:
            for k in GRADE_K:
                r = avaliar(casos, k, min_comuns, min_vizinhos)
                resultados.append(r)
                print(f"  {k:>3} {min_comuns:>7} {min_vizinhos:>7} {porcentagem(r['cobertura']):>7}"
                      f" {porcentagem(r['precisao']):>7} {porcentagem(r['recall']):>7}"
                      f" {porcentagem(r[f'precisao@{n}']):>7} {porcentagem(r[f'recall@{n}']):>7}"
                      f" {r['mae']:>6.3f} {r['rmse']:>6.3f}")

    with open(ARQUIVO_RESULTADOS, "w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(resultados[0].keys()))
        escritor.writeheader()
        escritor.writerows(resultados)
    print(f"\nResultados salvos em {ARQUIVO_RESULTADOS.name}")


if __name__ == "__main__":
    main()