# README

## How to use
Refer to `run.sh`, we split the experiments into 5 stages:

### Stage 0:
We extract all the waveform from `database-dir` to `output-dir`. (Note that downsampling is donw in this step)


```
python feature_extraction.py --database-dir "$database" --output-dir "$feat_dir"
```

### Stage 1:
In this stage, we train our svsnet with vcc2018.
```
python main.py --train-list 'list/sim_list_train.txt' --test-list 'list/sim_list_test.txt' --continue-from "$resume" \
                    --exp-dir "$exp_dir" --feat-path "$feat_dir" --model-type $model_type  --device cuda --batch-size 5 --dataset VCC2018
```

### Stage 2:
In this stage, we test on vcc2018;
```
python main.py  --test-list 'list/sim_list_test.txt' --continue-from $exp_dir/best_model.pt \
                    --exp-dir "$exp_dir" --feat-path "$feat_dir" --model-type $model_type --device cuda --batch-size 80 --testing --dataset VCC2018
```

### Stage 3:
In this stage, we extract feature from vcc2020.
```
python feature_extraction.py --database-dir "$database_vcc20" --output-dir "$feat_dir_vcc20"
```

### Stage 4:
In this stage, we test on vcc2020.
```
python main.py  --test-list 'list/vcc20_all_1.txt' --continue-from $exp_dir/best_model.pt \
                    --exp-dir $exp_dir_vcc20 --feat-path $feat_dir_vcc20 --model-type $model_type --device cuda --batch-size 80 --testing --dataset VCC2020
```

### Stage 5:
In this stage, we show all the result.
```
python evaluate.py $exp_dir/best_output.txt vcc2018
python evaluate.py $exp_dir_vcc20/best_output.txt vcc2020
```
