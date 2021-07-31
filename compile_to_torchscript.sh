#!/bin/bash
export MKL_THREADING_LAYER='GNU'

model="/home/madhav/styleposgan-stylegan2/checkpoints/"
src="/home/madhav/styleposgan-stylegan2/inf_test/source.jpg"
targ="/home/madhav/styleposgan-stylegan2/inf_test/target.jpg"
out='/home/madhav/styleposgan-stylegan2/inf_test/outputs'
scripted_model="/home/madhav/styleposgan-stylegan2/inf_test/scripted_model.pt"
densepose="/home/madhav/styleposgan-stylegan2/densepose"


# echo "python run_inference.py \
#     --save_model \
#     --model_dir $model \
#     --source $src \
#     --target $targ \
#     --outputs $out \
#     --scripted_model_path $scripted_model \
#     --densepose $densepose \
#     --image_size 256 -v \
#     "

python3.8 run_inference.py \
    --save_model \
    --model_dir $model \
    --source $src \
    --target $targ \
    --outputs $out \
    --scripted_model_path $scripted_model \
    --densepose $densepose \
    --image_size 256 -v