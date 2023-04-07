for i in 1 2 3 4 5
    do
        for LOSS in approxNDCG neuralNDCG rankNet rmse
            do
                python3 allrank/main.py --config-file-name configs/allRank/shallow/${LOSS}_config_MQ.json --run-id shallow_${LOSS}_MQ_${i} --job-dir test_logs/shallow_${LOSS}_MQ/fold_${i} --fold-dir Fold${i}
            done
    done