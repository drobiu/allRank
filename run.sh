for arch in deep shallow
    do
        for i in {1..5}
            do
                for LOSS in neuralNDCG rankNet rmse
                    do
                        python allrank/main.py --config-file-name configs/allRank/${arch}/${LOSS}_config_MQ.json --run-id ${arch}_${LOSS}_MQ_${i} --job-dir test_logs/MQ/${arch}_${LOSS}/fold_${i} --fold-dir Fold${i}
                    done
            done
    done