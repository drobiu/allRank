for arch in deep shallow
    do
        for LOSS in approxNDCG neuralNDCG rankNet rmse
            do
                python allrank/main.py --config-file-name configs/allRank/${arch}/${LOSS}_config_MSLR.json --run-id ${arch}_${LOSS}_MSLR_1 --job-dir test_logs/MSLR/${arch}_${LOSS}/fold_1 --fold-dir Fold1
            done
    done