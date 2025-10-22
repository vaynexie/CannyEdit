CUDA_VISIBLE_DEVICES=0 python main_cannyedit.py \
  --image_path './examples/girl.jpeg' \
  --image_whratio_unchange \
  --save_folder './results/girl_toboy_addmonkey.png' \
  --prompt_source "A young girl with red hair smiles brightly, wearing a red and white checkered shirt." \
  --prompt_local "A boy smiling." \
  --prompt_local "A monkey playing." \
  --mask_input "./examples/girl_mask_fg.png" \
  --mask_input "./examples/girl_mask_monkey.png" \
  --prompt_target "A young boy wearing a red and white checkered shirt, a monkey playing nearby."
