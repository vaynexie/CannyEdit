CUDA_VISIBLE_DEVICES=0 python main_cannyedit.py \
  --image_path './examples/twopeople.jpg' \
  --preserve_aspect_ratio \
  --save_folder './results/twopeople_remove.png'  \
  --prompt_local '[remove]' \
  --dilate_mask \
  --mask_input "./examples/twopeople_mask.png" \
  --true_gs 4 \
  --seed 9539396
