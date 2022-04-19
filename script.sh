ETA=("0.05" "0.1" "0.2" "0.5")
MAX_DEPTH=(1 2 5 10)

for eta in ${ETA[@]}
do
    for max_depth in ${MAX_DEPTH[@]}
    do
        python3 barra_CNN_novo.py $eta $max_depth
    done
done