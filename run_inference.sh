#!/bin/bash
#Local Machine Only
export MKL_THREADING_LAYER='GNU'

model="/home/madhav/styleposgan-stylegan2/checkpoints/models/256-5e5"
src="/home/madhav/styleposgan-stylegan2/res_test/lowres_df.jpg"
targ="/home/madhav/styleposgan-stylegan2/res_test/highres_df.jpg"
out='/home/madhav/styleposgan-stylegan2/res_test/outputs'
scripted_model="/home/madhav/styleposgan-stylegan2/checkpoints/scripted_model_1.0.2.pt"
densepose="/home/madhav/styleposgan-stylegan2/densepose"


python3.8 run_inference_prod.py \
    --source $src \
    --target $targ \
    --outputs $out \
    --scripted_model_path $scripted_model \
    --densepose $densepose \
    --image_size 256 -v \
    # --model_dir $model  \
    # --save_model 
