#!/bin/bash
embedding_names="glovevector/glove.6B.100d.txt glovevector/glove.6B.50d.txt glovevector/glove.6B.200d.txt glovevector/glove.6B.300d.txt random word2vec"

for units in 128  
do
    for num_layers in 2 
    do
        for seed in `seq 0 1 2`  
        do
            for embedding_name in $embedding_names 
            do 
                #for trainable in 0 1 
                #do 
                    for window_size in 0 1 2 3
                    do 
                        for dim in 50 100 200 300 
                        do 
                            echo "units $units, num_layers $num_layers, seed $seed, embedd_name $embedding_name, trainable $trainable, window_size $window_size, embedd_dim $dim"
                            python tf_pos_lstm_main.py train -m 20 -u $units -N $num_layers -E $seed -e $embedding_name -t 1 -w $window_size -d $dim -s -1 
                        done 
                    done 
                #done
            done
        done
    done
done

