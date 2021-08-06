for opt in 0 1 2
do
    for pnum in 2 3 4 5 6
    do
        for testnum in 11 12 13 14 15
        do
            python3 segmeninf.py $opt 2 $pnum $testnum 
        done
    done
done