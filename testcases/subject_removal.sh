CUDA_VISIBLE_DEVICES=0 python main_cannyedit.py \
  --image_path './examples/girl.jpeg' \
  --image_whratio_unchange \
  --save_folder './results/girl_remove.png' \
  --prompt_local '[remove]' \
  --mask_input "./examples/girl_mask_fg.png" \
  --dilate_mask
