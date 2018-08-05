for n in 100 250 500
do
    for m in 1 2 3 4 5
    do
        for k in 1 2 5 7 9
        do
            M=$(($m*$n))
            K=$(($k*$n/10))
            echo "$n, $M, $K"
            python3 PythonImplementation/generateInput.py $n $M $K 50 > Input_n_m_k_iters/input.$n.$M.$K.50.txt
        done
    done
done
