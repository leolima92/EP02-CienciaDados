"""Leitura dos CSVs do MovieLens usando apenas a biblioteca padrão."""
import csv
import re
from pathlib import Path

PASTA_DADOS = Path(__file__).parent / "data"
ANO_NO_TITULO = re.compile(r"\((\d{4})\)\s*$")


def ler_filmes(caminho=PASTA_DADOS / "movies.csv"):
    """Devolve uma lista de dicts: movieId, title, year, genres (lista)."""
    filmes = []
    with open(caminho, encoding="utf-8", newline="") as arquivo:
        for linha in csv.DictReader(arquivo):
            titulo = linha["title"].strip()
            ano = ANO_NO_TITULO.search(titulo)
            generos = linha["genres"].split("|")
            if generos == ["(no genres listed)"]:
                generos = []
            filmes.append({
                "movieId": int(linha["movieId"]),
                "title": titulo,
                "year": int(ano.group(1)) if ano else None,
                "genres": generos,
            })
    return filmes


def ler_avaliacoes(caminho=PASTA_DADOS / "ratings.csv"):
    """Devolve uma lista de dicts: userId, movieId, rating, timestamp."""
    avaliacoes = []
    with open(caminho, encoding="utf-8", newline="") as arquivo:
        for linha in csv.DictReader(arquivo):
            avaliacoes.append({
                "userId": int(linha["userId"]),
                "movieId": int(linha["movieId"]),
                "rating": float(linha["rating"]),
                "timestamp": int(linha["timestamp"]),
            })
    return avaliacoes


if __name__ == "__main__":
    filmes = ler_filmes()
    avaliacoes = ler_avaliacoes()
    generos = {g for f in filmes for g in f["genres"]}
    usuarios = {a["userId"] for a in avaliacoes}

    print(f"Filmes:     {len(filmes)}")
    print(f"Avaliações: {len(avaliacoes)}")
    print(f"Usuários:   {len(usuarios)}")
    print(f"Gêneros:    {len(generos)} -> {sorted(generos)}")
    print(f"Exemplo:    {filmes[0]}")