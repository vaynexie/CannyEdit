CUDA_VISIBLE_DEVICES=0 python main_cannyedit.py \
  --image_path './examples/girl.jpeg' \
  --preserve_aspect_ratio \
  --save_folder './results/girl_remove.png' \
  --prompt_local '[remove]' \
  --mask_input "./examples/girl_mask_fg.png" \
  --dilate_mask
