export CUDA_VISIBLE_DEVICES=3

python main.py \
--output './results' \
--style './style_images' \
--sty_guidance 1.15 \
--num_inference_steps 6 \
--cfg_guidance 7.0 \
--fix_step_index 200 \
--start_ac_layer 7 \
--end_ac_layer 16