import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--num-iter-boost",
    "--n",
    help="Número de iterações no boosting. Relevante apenas nos testes com o xgb.train. Argumento relevante apenas na validação",
    type=int,
    default=10,
)
parser.add_argument(
    "--taxa-apren",
    "-t",
    help="Taxa de aprendizado. Argumento relevante apenas na validação",
    type=float,
    default=0.3,
)
parser.add_argument(
    "--prof-max",
    "-p",
    help="Profundidade máxima. Argumento relevante apenas na validação",
    type=int,
    default=6,
)
parser.add_argument(
    "--rotulo",
    "-r",
    help="Qual fluxo será prevista pelo classificador: barra_fluxo ou costa_fluxo",
    type=str,
    default="barra_fluxo",
)
parser.add_argument(
    "--iteracao",
    "-i",
    help="O número da iteração. Irrelevante na validação",
    type=int,
    default=1,
)
parser.add_argument(
    "--tam-passo",
    "-s",
    help="Número de registros que será fornecido para o classificador de uma vez. Relevante apenas para o xgbtrain-6em6 e xgbreg-6em6",
    type=int,
    default=6,
)
parser.add_argument(
    "--validacao",
    help="Executar validação. Omita esse argumento para realizar testes.",
    action="store_true",
)
parser.add_argument(
    "--nexec-xgbtrain-iter",
    help="Não executar xgb.train com treinamento iterativo",
    action="store_true",
)
parser.add_argument(
    "--nexec-xgbtrain-tudo",
    help="Não executar xgb.train com treinamento passando a base de treinamento tudo de uma vez só",
    action="store_true",
)
parser.add_argument(
    "--nexec-xgbreg-iter",
    help="Não executar XGBRegressor com treinamento iterativo",
    action="store_true",
)
parser.add_argument(
    "--nexec-xgbreg-tudo",
    help="Não executar XGBRegressor com treinamento passando a base de treinamento tudo de uma vez só",
    action="store_true",
)

argumentos = parser.parse_args()

hiperparametros = {
    "num_boost_round": argumentos.num_iter_boost,
    "taxa_de_aprendizado": argumentos.taxa_apren,
    "profundidade_maxima": argumentos.prof_max,
}
