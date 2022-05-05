LABELS=("barra_fluxo" "costa_fluxo")
N_BOOST_ROUND=(1 10)
ETA=("0.05" "0.1" "0.2" "0.5")
MAX_DEPTH=(1 2 5 10)
ITERACOES=10

for label in ${LABELS[@]}
do
    echo "Realizando validação na $label"
    for boost_round in ${N_BOOST_ROUND[@]}
    do
        for eta in ${ETA[@]}
        do
            for max_depth in ${MAX_DEPTH[@]}
            do
                python3 barra_CNN_novo.py $boost_round $eta $max_depth $label 0 --validacao 2> erro
            done
        done
    done
    echo "Realizando testes"
    for x in $(seq $ITERACOES)
    do
        python3 barra_CNN_novo.py 0 0 0 $label $x 2> erro
    done
done