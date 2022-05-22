LABELS=("barra_fluxo" "costa_fluxo")
N_BOOST_ROUND=(10 20 30)
ETA=("0.05" "0.1" "0.2" "0.5")
MAX_DEPTH=(1 2 5 10)
ITERACOES=10
PASSOS=6

for label in ${LABELS[@]}
do
    echo "Realizando validação na $label"
    for boost_round in ${N_BOOST_ROUND[@]}
    do
        for eta in ${ETA[@]}
        do
            for max_depth in ${MAX_DEPTH[@]}
            do
                python3 barra_CNN_novo.py --num-iter-boost=$boost_round --taxa-apren=$eta --prof-max=$max_depth --rotulo=$label --tam-passo=PASSOS --validacao 2> erro
            done
        done
    done
    echo "Realizando testes"
    for x in $(seq $ITERACOES)
    do
        python3 barra_CNN_novo.py --rotulo=$label --iteracao=$x --tam-passo=PASSOS 2> erro
    done
done