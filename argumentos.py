import sys

num_boost_round = int(sys.argv[1])
taxa_de_aprendizado = float(sys.argv[2])
profundidado_maxima = int(sys.argv[3])
label = sys.argv[4]
iteracao = int(sys.argv[5])
validacao = "--validacao" in sys.argv
nao_executar_xgbtrain_6em6 = "--nexec-xgbtrain-6em6" in sys.argv
nao_executar_xgbtrain_tudo = "--nexec-xgbtrain-tudo" in sys.argv
nao_executar_xgbreg_6em6 = "--nexec-xgbreg-6em6" in sys.argv
nao_executar_xgbreg_tudo = "--nexec-xgbreg-tudo" in sys.argv

hiperparametros = {
    "num_boost_round": num_boost_round,
    "taxa_de_aprendizado": taxa_de_aprendizado,
    "profundidade_maxima": profundidado_maxima,
}
