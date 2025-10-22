CUDA_VISIBLE_DEVICES=0 python main_cannyedit.py \
  --image_path './examples/girl.jpeg' \
  --image_whratio_unchange \
  --save_folder './results/girl_to_boy.png' \
  --prompt_local "A boy smiling." \
  --prompt_source "A young girl with red hair smiles brightly, wearing a red and white checkered shirt." \
  --prompt_target "A young boy with red hair smiles brightly, wearing a red and white checkered shirt." \
  --mask_input "./examples/girl_mask_fg.png"
