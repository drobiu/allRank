for i in {1..5}
    do
        for LOSS in approxNDCG neuralNDCG rankNet rmse
            do
                python allrank/main.py --config-file-name configs/allRank/deep/${LOSS}_config_MQ.json --run-id deep_${LOSS}_MSLR_${i} --job-dir test_logs/deep_${LOSS}_MSLR/fold_${i} --fold-dir Fold${i}
            done
    done