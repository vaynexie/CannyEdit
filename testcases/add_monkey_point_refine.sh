CUDA_VISIBLE_DEVICES=0 python main_cannyedit.py \
 --image_path './examples/girl.jpeg' \
 --image_whratio_unchange \
 --save_folder './results/girl_add_monkey_point/' \
 --prompt_local "A monkey playing." \
 --prompt_source "A young girl with red hair smiles brightly, wearing a red and white checkered shirt." \
 --prompt_target "A young girl with red hair smiles brightly, wearing a red and white checkered shirt, a monkey playing nearby." \
 --refine_mask \
 --mask_input "(0.12,0.35)"
