database="../../dataset/VCC2018/wav/"
feat_dir="../../feature/test_vcc2018_for_delete/wav"
exp_dir="result/VCC2018"

database_vcc20="../../dataset/VCC2020/"
feat_dir_vcc20="../../feature/test_VCC2020_for_delete/wav/"
exp_dir_vcc20="result/VCC2020"

model_type="regression"
resume=""

stage=5
stop_stage=5
set -u

if [ "$stage" -le "0" ] && [ "$stop_stage" -ge "0" ]; then
    python feature_extraction.py --database-dir "$database" --output-dir "$feat_dir"
fi

# training or resuming
if [ "$stage" -le "1" ] && [ "$stop_stage" -ge "1" ]; then
    python main.py --train-list 'list/sim_list_train.txt' --test-list 'list/sim_list_test.txt' --continue-from "$resume" \
                    --exp-dir "$exp_dir" --feat-path "$feat_dir" --model-type $model_type  --device cuda --batch-size 5 --dataset VCC2018 --epoch 1
fi

# testing VCC18
if [ "$stage" -le "2" ] && [ "$stop_stage" -ge "2" ]; then
    python main.py  --test-list 'list/sim_list_test.txt' --continue-from $exp_dir/best_model.pt \
                    --exp-dir "$exp_dir" --feat-path "$feat_dir" --model-type $model_type --device cuda --batch-size 80 --testing --dataset VCC2018
fi

if [ "$stage" -le "3" ] && [ "$stop_stage" -ge "3" ]; then
    python feature_extraction.py --database-dir "$database_vcc20" --output-dir "$feat_dir_vcc20" 
fi

# testing VCC18
if [ "$stage" -le "4" ] && [ "$stop_stage" -ge "4" ]; then
    python main.py  --test-list 'list/vcc20_all_1.txt' --continue-from $exp_dir/best_model.pt \
                    --exp-dir $exp_dir_vcc20 --feat-path $feat_dir_vcc20 --model-type $model_type --device cuda --batch-size 80 --testing --dataset VCC2020
fi

# showing result
if [ "$stage" -le "5" ] && [ "$stop_stage" -ge "5" ]; then
    echo "VCC2018:"
    python evaluate.py $exp_dir/best_output.txt vcc2018
    echo "VCC2020:"
    python evaluate.py $exp_dir_vcc20/best_output.txt vcc2020
fi
