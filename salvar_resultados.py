import os

import pandas as pd
import argumentos
import json


# Config
DIR_RESULTADOS = "resultados"
CAMINHO_RESULTADOS = "resultados"

COLUNAS = [
    "mae",
    "mse",
    "rmse",
    "r2",
    "tempo_train",
    "tempo_teste",
    "tempo_total",
]


def salvar_resultados(resultados, nome_classificador):
    caminho_resultado = (
        f"{CAMINHO_RESULTADOS}/{nome_classificador}_{argumentos.label}.csv"
    )
    # Carregar csv
    dados_csv = {}
    if os.path.exists(caminho_resultado):
        dados_csv = pd.read_csv(caminho_resultado, index_col=0).to_dict()

    indice = int(argumentos.iteracao)

    for resultado in COLUNAS:
        nome_coluna = resultado
        if not nome_coluna in dados_csv:
            dados_csv[nome_coluna] = {}

        dados_csv[nome_coluna][indice] = float(resultados[resultado])

    pd.DataFrame(data=dados_csv).to_csv(caminho_resultado, index_label="x")


def salvar_resultados_validacao(resultados):
    caminho_resultado = f"{CAMINHO_RESULTADOS}/{argumentos.label}.json"
    # Carregar json
    dados_json = []
    if os.path.exists(caminho_resultado):
        with open(caminho_resultado, "r") as file:
            dados_json = json.load(file)
    registro = {}
    for nomeh, valorh in argumentos.hiperparametros.items():
        registro[nomeh] = valorh
    for nomer, valorr in resultados.items():
        registro[nomer] = valorr
    dados_json.append(registro)

    with open(caminho_resultado, "w") as file:
        file.write(json.dumps(dados_json))


# Criar diretórios
if not os.path.isdir(DIR_RESULTADOS):
    os.mkdir(DIR_RESULTADOS)
if not os.path.isdir(CAMINHO_RESULTADOS):
    os.mkdir(CAMINHO_RESULTADOS)
