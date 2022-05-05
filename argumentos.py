import sys

num_boost_round = int(sys.argv[1])
taxa_de_aprendizado = float(sys.argv[2])
profundidado_maxima = int(sys.argv[3])
label = sys.argv[4]
iteracao = int(sys.argv[5])
validacao = "--validacao" in sys.argv

hiperparametros = {
    "num_boost_round": num_boost_round,
    "taxa_de_aprendizado": taxa_de_aprendizado,
    "profundidade_maxima": profundidado_maxima,
}
