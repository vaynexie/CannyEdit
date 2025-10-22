CUDA_VISIBLE_DEVICES=0  python main_cannyedit.py \
--image_path './examples/bager1.png' \
--image_whratio_unchange \
--save_folder './results/bager_add_manwoman_auto_inferpoint/' \
--prompt_local "a man customer reading menu." \
--prompt_local "a woman waiter ready to serve." \
--prompt_source 'A cozy restaurant entrance.' \
--prompt_target 'A cozy restaurant entrance with a man reading the menu while a woman waiter stands ready to serve.' \
--refine_mask \
--self_infer_point --seed 3817513
