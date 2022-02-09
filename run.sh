exp_dir="result/SVSNet"
feat_dir="../../feature/SVSNet"
model_type="regression"
resume=""
stage=1
stop_stage=1

set -x
set -u

# training or resuming
if [ "$stage" -le "1" ] && [ "$stop_stage" -ge "1" ]; then
    python main.py --train-list '../../feature/sim_net/sim_list_train.txt' --test-list '../../feature/sim_net/sim_list_test.txt' --continue-from "$resume" \
                    --exp-dir $exp_dir --feat-path $feat_dir --model-type $model_type --epoch 1 --device cuda --batch-size 5
fi

# testing VCC18
if [ "$stage" -le "2" ] && [ "$stop_stage" -ge "2" ]; then
    python main.py --train-list '../../feature/sim_net/sim_list_train.txt' --test-list '../../feature/sim_net/sim_list_test.txt' --continue-from $exp_dir/best_model.pt \
                    --exp-dir $exp_dir --feat-path $feat_dir --model-type $model_type --epoch 1 --device cuda --batch-size 80 --testing
fi


exp_dir_vcc20="result/VCC20"
feat_dir_vcc20="../../feature/VCC20"

# testing VCC18
if [ "$stage" -le "4" ] && [ "$stop_stage" -ge "4" ]; then
    python main.py - --test-list '../../dataset/VCC20/VCC2020-listeningtest-info/VCC202-listeningtest-scores/vcc20_all.txt' --continue-from $exp_dir/best_model.pt \
                    --exp-dir $exp_dir_vcc20 --feat-path $feat_dir_vcc20 --model-type $model_type --epoch -1 --device cuda --batch-size 80 --testing
fi

# showing result
if [ "$stage" -le "5" ] && [ "$stop_stage" -ge "5" ]; then
    python logger.py --exp-dir $exp_dir
fi