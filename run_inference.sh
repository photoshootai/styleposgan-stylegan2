#!/bin/bash
#Local Machine Only
export MKL_THREADING_LAYER='GNU'

model="/home/madhav/styleposgan-stylegan2/checkpoints/"
src="/home/madhav/styleposgan-stylegan2/inf_test/source.jpg"
targ="/home/madhav/styleposgan-stylegan2/inf_test/target.jpg"
out='/home/madhav/styleposgan-stylegan2/inf_test/outputs'
scripted_model="/home/madhav/styleposgan-stylegan2/checkpoints/scripted_model_1.0.0.pt"
densepose="/home/madhav/styleposgan-stylegan2/densepose"

python3.8 run_inference_prod.py \
    --source $src \
    --target $targ \
    --outputs $out \
    --scripted_model_path $scripted_model \
    --densepose $densepose \
    --image_size 256 -v