CUDA_VISIBLE_DEVICES=0 python main_cannyedit.py \
  --image_path './examples/girl.jpeg' \
  --image_whratio_unchange \
  --save_folder './results/girl_bg.png' \
  --prompt_local "A mountain." \
  --prompt_source "A young girl with red hair smiles brightly, wearing a red and white checkered shirt." \
  --prompt_target "A young girl with red hair smiles brightly, wearing a red and white checkered shirt, sitting on a bench with mountains in the background." \
  --mask_input "./examples/girl_mask_bg.png"
