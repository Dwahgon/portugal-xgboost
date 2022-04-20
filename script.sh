N_BOOST_ROUND=(1 10)
ETA=("0.05" "0.1" "0.2" "0.5")
MAX_DEPTH=(1 2 5 10)
N_ESTIMATORS=(1 30 50 100)

for boost_round in ${N_BOOST_ROUND[@]}
do
    for eta in ${ETA[@]}
    do
        for max_depth in ${MAX_DEPTH[@]}
        do
            # for n_estimators in ${N_ESTIMATORS[@]}
            # do
            python3 barra_CNN_novo.py $boost_round $eta $max_depth $n_estimators
            # done
        done
    done
done