import os
import argparse
import torch
import torch.nn.functional as F
from src.flux.xflux_pipeline_cannyedit import XFluxPipeline
from PIL import Image, ImageColor, ImageDraw, ImageFont
import cv2
from scipy.ndimage import binary_dilation
import numpy as np
import copy
import re
import random
import itertools
import copy
import time
from glob import glob
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import gc
import sys
import matplotlib
import ast
import warnings
warnings.filterwarnings("ignore")
# from huggingface_hub import login
# login(token="YOUR_HUGGINGFACE_ACCESS_TOKEN")




def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipe", default=None)
    parser.add_argument(
        "--sam_checkpoint", default='./model_checkpoints/sam2.1_hiera_large.pt')
    parser.add_argument(
        "--gdino_checkpoint", default='./model_checkpoints/grounding-dino-base')
    parser.add_argument(
        "--qwen_checkpoint_path", default='./model_checkpoints/Qwen2.5-VL-7B-Instruct')
    parser.add_argument(
        "--qwen_llm_checkpoint_path", default='./model_checkpoints/Qwen3-4B-Instruct-2507')
    parser.add_argument(
        "--internvl_checkpoint_path", default='./model_checkpoints/InternVL3-14B')
    parser.add_argument(
        "--prompt_source", type=str,  # required=True,
        help="The text prompt that describes the source image"
    )
    parser.add_argument(
        "--prompt_target",  # required=True,
        help="The text prompt that describes the targeted image after editing"
    )
    parser.add_argument(
        "--prompt_local",
        action="append",
        help="The local prompt(s) for edit region(s)",
    )
    parser.add_argument(
        "--mask_input",
        action="append",
        help="path(s) of mask(s) or tuple(s) of point(s) [in '(x,y)', the value is between 0 and 1, (0,0)-topmost leftmost, (1,1)-bottom-most right-most] indicating the region to edit",
    )
    parser.add_argument(
        "--norm_softmask",
        default=True,
        help="Apply normalization on softmask in the point-hint setting"
    )
    parser.add_argument(
        "--dilate_mask", action='store_true',
        help="Dilate the mask"
    )
    parser.add_argument(
        "--fill_hole_mask",
        action='store_true',
        default=True,
        help="Fill the holes in the mask, useful for the imprecise segmentation masks"
    )

    parser.add_argument(
        "--width", type=int, default=768, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=768, help="The height for generated image"
    )
    parser.add_argument(
        "--image_whratio_unchange", action='store_true',
        help="In default we use square input/output, set this to True if you wish to keep the original image width/height ratio unchanged."
    )
    parser.add_argument(
        "--save_folder", type=str, default='./cannyedit_outputs/', help="Folder to save"
    )
    parser.add_argument(
        "--neg_prompt2", type=str,
        default="focus,centered foreground, humans, objects, noise, blurring, low resolution, artifacts, distortion, overexposure, and uneven lighting, bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face,  disconnected limbs",#, #
        help="The input text negative prompt2"
    )
    parser.add_argument(
        "--neg_prompt", type=str,
        default="humans, objects, noise, blurring, low resolution, artifacts, distortion, overexposure, and uneven lighting, bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face,  disconnected limbs",#, #
        help="The input text negative prompt"
    )
    parser.add_argument(
        "--control_weight2", type=float, default=0.7, help="Controlnet model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--refine_mask", action='store_true',
        help="If true, we will cache the inversion result and previous generation result, and then allow the editing with mask refinement"
    )
    parser.add_argument(
        "--multi_run", action='store_true',
        help="If true, we will cache the inversion result and previous generation result, and then allow the multi-run edits"
    )
    parser.add_argument(
        "--self_infer_point",action='store_true',
        help="If true, we will use InternVL3 to infer to point hints for objects to add"
    )
    parser.add_argument(
        "--inversion_save_path", type=str, default=None, help="Path to save the inversion result"
    )
    parser.add_argument(
        "--generate_save_path", type=str, default=None, help="Path to save the previous generation result"
    )
    parser.add_argument(
        "--generate_next_round_flag", default=False
    )

    parser.add_argument(
        "--auto_mask_refine", action='store_true', help="auto mask refinement via language SAM"
    )
    parser.add_argument(
        "--img_prompt", type=str, default=None,
        help="Path to input image prompt"
    )
    parser.add_argument(
        "--neg_img_prompt", type=str, default=None,
        help="Path to input negative image prompt"
    )
    parser.add_argument(
        "--ip_scale", type=float, default=1.0,
        help="Strength of input image prompt"
    )
    parser.add_argument(
        "--neg_ip_scale", type=float, default=1.0,
        help="Strength of negative input image prompt"
    )
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (Controlnet)"
    )

    parser.add_argument(
        "--name", type=str, default=None,
        help="A filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (IP-Adapter)"
    )
    parser.add_argument(
        "--ip_name", type=str, default=None,
        help="A IP-Adapter filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_local_path", type=str, default=None,
        help="Local path to the model checkpoint (IP-Adapter)"
    )
    parser.add_argument(
        "--lora_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (LoRA)"
    )
    parser.add_argument(
        "--lora_name", type=str, default=None,
        help="A LoRA filename to download from HuggingFace"
    )
    parser.add_argument(
        "--lora_local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", default=False, help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--use_ip", action='store_true', help="Load IP model"
    )
    parser.add_argument(
        "--use_lora", action='store_true', help="Load Lora model"
    )
    parser.add_argument(
        "--use_controlnet", action='store_true', help="Load Controlnet model"
    )
    parser.add_argument(
        "--use_paint", action='store_true', help="Load inpainting model"
    )
    parser.add_argument(
        "--image_path", type=str, default=None, help="Path to image"
    )

    parser.add_argument(
        "--lora_weight", type=float, default=0.9, help="Lora model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_weight", type=float, default=0.8, help="Controlnet model strength (from 0 to 1.0)"
    )

    parser.add_argument(
        "--control_type", type=str, default="canny",
        choices=("canny", "openpose", "depth", "zoe", "hed", "hough", "tile"),
        help="Name of controlnet condition, example: canny"
    )
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )

    parser.add_argument(
        "--num_steps", type=int, default=50, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=random.randint(0, 9999999), help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=2, help="true guidance"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=5, help="timestep to start true guidance"
    )

    parser.add_argument(
        "--paint_task", type=str, default=None, help="The task for inpainting"
    )
    parser.add_argument(
        "--input_image_path", type=str, default=None, help="Path to image for painting"
    )
    parser.add_argument(
        "--input_mask_path", type=str, default=None, help="Path to mask for painting"
    )
    return parser


# =======================================================================================================================

def process_mask(mask_path,
                 height,
                 width,
                 dilate=False,
                 dilation_kernel_size=(5, 5),
                 fill_holes=False,
                 closing_kernel_size=(5, 5)):
    """
    Processes a mask image, optionally fills holes, dilates it, and returns a simple mask tensor.

    Args:
        mask_path (str): The path to the mask image.
        height (int): The desired height for the original image dimensions.
        width (int): The desired width for the original image dimensions.
        dilate (bool, optional): If True, performs a dilation operation to expand the
                                 mask area. Defaults to False.
        dilation_kernel_size (tuple, optional): The size of the kernel for dilation.
                                                Defaults to (5, 5).
        fill_holes (bool, optional): If True, performs a morphological closing operation
                                     to fill small holes within the mask. Defaults to False.
        closing_kernel_size (tuple, optional): The size of the kernel for the closing
                                               operation. Defaults to (5, 5).

    Returns:
        torch.Tensor: The processed simple mask tensor, ready for use.
    """
    # Read the mask image
    mask = cv2.imread(mask_path)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask file from: {mask_path}")

    # Convert the mask to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Downsample the mask to 1/16th of the target dimensions
    downsampled_mask = cv2.resize(mask, (width // 16, height // 16), interpolation=cv2.INTER_AREA)

    # Threshold the downsampled mask to a binary format (0 or 255)
    _, binary_downsampled_mask = cv2.threshold(downsampled_mask, 127, 255, cv2.THRESH_BINARY)

    # --- Optional Hole Filling Step ---
    # This operation is ideal for making masks contiguous and removing "pepper" noise.
    if fill_holes:
        # Create the kernel for the closing operation.
        kernel = np.ones(closing_kernel_size, np.uint8)
        # Apply morphological closing.
        binary_downsampled_mask = cv2.morphologyEx(binary_downsampled_mask, cv2.MORPH_CLOSE, kernel)

    # --- Optional Dilation Step ---
    # This expands the outer boundary of the mask.
    if dilate:
        # Create a kernel for the dilation.
        kernel = np.ones(dilation_kernel_size, np.uint8)
        # Apply the dilation operation.
        binary_downsampled_mask = cv2.dilate(binary_downsampled_mask, kernel, iterations=1)

    # Normalize the binary mask to have values of 0 and 1
    binary_downsampled_mask = (binary_downsampled_mask // 255).astype(np.uint8)

    # Invert the mask (object area becomes 0, background becomes 1)
    local_mask = 1 - binary_downsampled_mask

    # Convert the final mask to a PyTorch tensor
    local_mask_tensor = torch.tensor(local_mask, dtype=torch.float32)

    return local_mask_tensor


def plot_image_with_mask(image_path, mask_path_list, width, height, save_path):
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image to the specified width and height
    image = cv2.resize(image, (width, height))

    # Convert the image from BGR to RGB for proper display in matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis("off")

    for mask_path in mask_path_list:
        # Load the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize the mask to match the resized image dimensions
        mask = cv2.resize(mask, (width, height))

        # Find the coordinates of the white region in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get the bounding rectangle for each contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw the bounding rectangle as a red box
            rect = plt.Rectangle((x, y), w, h, edgecolor='red', fill=False, linewidth=4)
            ax.add_patch(rect)

    # Save the figure to the specified output path
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return save_path





def visualize_with_star(image_path, center_x, center_y):
    """
    Visualize the point in the image with a yellow star.

    Args:
        image_path (str): Path to the input image.
        center_x (float): X-coordinate of the center (normalized between 0 and 1).
        center_y (float): Y-coordinate of the center (normalized between 0 and 1).
        output_path (str): Path to save the output image with the star.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Convert the image from BGR to RGB for Matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the pixel coordinates of the center
    center_pixel_x = int(center_x * width)
    center_pixel_y = int(center_y * height)

    # Draw a yellow star at the center location
    star_size = 30  # Radius of the star
    cv2.drawMarker(
        image,
        (center_pixel_x, center_pixel_y),  # Center of the star
        color=(255, 255, 0),  # Yellow color in RGB
        markerType=cv2.MARKER_STAR,
        markerSize=star_size,
        thickness=2,
    )

    # Display the image with Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Image with Yellow Star")
    plt.show()

# ============================================================================================================================

def main(args):
    ##===================Config===============================================================================================
    ## flag for deciding if the edit is for removal
    removal_flag=False
    ## flag for deciding if the edit is to use the point hints for addition
    point_flag=0
    ## flag for deciding whether to regenerate the outputs before mask refinement
    point_regenerate=False

    mask_path_list=[]
    image = Image.open(args.image_path).convert("RGB")
    image_name=args.image_path.split('/')[::-1][0].split('.')[0]
    os.makedirs("./mask_temp/", exist_ok=True)


    ## args.image_whratio_unchange == True: if true, keep the image height/width ratio unchanged
    if args.image_whratio_unchange == True:
        widtho, heighto = image.size
        maxone = np.max([widtho, heighto])
        if maxone == widtho:
            args.width = args.width
            args.height = int(args.width * (heighto / widtho))
        else:
            args.height = args.height
            args.width = int(args.height * (widtho / heighto))
        print('Keep image width/height ratio unchanged, we now use：[width, height]=' + str([args.width, args.height]))

    if args.pipe != None:
        xflux_pipeline = args.pipe
    else:
        xflux_pipeline = XFluxPipeline(args.model_type, args.device, args.offload)
        xflux_pipeline.set_controlnet(args.control_type, args.local_path, args.repo_id, args.name)
    ##===================END of Config========================================================================================

    ##==============Prepare Prompts (local edit prompts, source prompt, target prompt)========================================
    ## Have the user input the local prompts if they are not provided in the code line
    if args.prompt_local == None:
        args.prompt_local = []
        print('No local prompt provided.')
        args.prompt_local.append(input("Enter the first local prompt (enter [remove] if wish to remove objects): "))
        for kk in range(10):
            resp = input(
                "Enter the next local prompt if you may have (enter [remove] if wish to remove objects), enter 'done' if you have finished all inputs: ")
            if resp == "done":
                break
            else:
                args.prompt_local.append(resp)

    ## The local prompt for the object to be removed is default to be "empty background"+' out-of-focus, atmospheric background'
    for pp_ind in range(len(args.prompt_local)):
        if "[remove]" in args.prompt_local[pp_ind]:
            args.prompt_local[pp_ind]="empty background"+' out-of-focus, atmospheric background'
            removal_flag=True

    ##Apply vlm to generate source  prompt and target prompt if not provided
    ##use vlm to get source prompt and target prompt
    if args.prompt_source == None or args.prompt_target == None:
        print('no source/target prompt is provided, using QWEN2.5-VL to generate the prompt automatically \n')

        from helper.qwen25 import process_vision_info
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.qwen_checkpoint_path, torch_dtype="auto",
                                                                        device_map="auto")
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28
        qwen_processor = AutoProcessor.from_pretrained(args.qwen_checkpoint_path, min_pixels=min_pixels,
                                                       max_pixels=max_pixels)
        if args.prompt_source == None:
            from helper.qwen25 import generate_output_by_qwen
            output_text= generate_output_by_qwen(qwen_model, qwen_processor,  args.image_path, args.height, args.width, "Describe this image in 15 words.", max_new_tokens=128)
            args.prompt_source = output_text[0]

            print('\n')
            print('VLM generated source prompt: ' + args.prompt_source)
            print('\n')
        if args.prompt_target == None:
            print(
                'Important: Currently the auto generation of target prompts only support **adding and removal**. If the editing involves only edit tasks like replacement, please provide the target prompt here, you may refer to the VLM-generated source prompt.\n')
            if removal_flag == False:
                resp = input(
                    "Press '1' for using VLM to geernate target prompt, other enter the target prompt directly: \n")
            elif removal_flag == True:
                resp = '1'

            if resp != '1':
                args.prompt_target = resp
                print('Entered target prompt: ' + args.prompt_target)
                print('\n')
            if resp == '1':
                from helper.qwen25 import generate_output_by_qwen
                if removal_flag==False:
                    prompt_for_target = 'Given the caption for this image:' + str(
                        args.prompt_source) + 'Suppose there would be new objects in the image:'
                    words_count = 15
                    for object_add in args.prompt_local:
                        prompt_for_target += object_add + '; and '
                        words_count += 5
                    prompt_for_target += '\n Based on the original caption and the description to the new objects. Generate the new caption after the objects are added in ' + str(
                        words_count) + ' words.'  # Keep the original caption if possible'
                    output_text=generate_output_by_qwen(qwen_model, qwen_processor, args.image_path, args.height, args.width,
                                            prompt_for_target, max_new_tokens=128)

                    args.prompt_target = output_text[0]
                    print('VLM generated target prompt: ' + args.prompt_target)
                    print('\n')

                elif removal_flag == True:
                    args.prompt_target=' '
                    save_path=plot_image_with_mask(args.image_path, mask_path_list,width=args.width,height=args.height,save_path='./mask_temp/masktpimage.png')
                    output_text = generate_output_by_qwen(qwen_model, qwen_processor, save_path, args.height,
                                                          args.width,
                                                          "Support the objects within the red bounding box will be removed. Describe the image background excluding the removed objects in 10 words.", max_new_tokens=128)

                    args.prompt_target = output_text[0]
                    print('\n')
                    print('VLM generated target prompt for the removal task: ' + args.prompt_target)
                    print('\n')

                    output_text = generate_output_by_qwen(qwen_model, qwen_processor, save_path, args.height,
                                                          args.width,
                                                          "describe the region within the red bounding box in 10 words.",
                                                          max_new_tokens=128)
                    args.neg_prompt = output_text[0]
                    print('\n')
                    print('VLM generated negative prompt for the removal task: ' + output_text[0])
                    print('\n')

        del qwen_model
        del qwen_processor
    ##==============END of Prepare Prompts ===================================================================================

    ##==============Prepare Masks=============================================================================================
    ### Read the mask files or create the masks online
    '''
    CannyEdit support the input of binary masks or point hints as indicators of locations to edit
    User can either input the paths of the binary mask image or the list of points in (x,y) where the x,y values are between 0 and 1.
    '''
    ## Read the mask files is provided
    if args.mask_input != None:
        parsed_inputs = []
        ## mask_input: either mask paths of list of points
        ## if the mask_input include list of points, then turn the point_flag to 1
        for item in args.mask_input:
            try:
                parsed = ast.literal_eval(item)
                point_flag=1
                parsed_inputs.append(parsed)
            except (SyntaxError, ValueError):
                parsed_inputs.append(item)  # fallback: keep as string if not a tuple
    ## if point_flag==0, read the masks from the mask paths
    if args.mask_input != None and point_flag==0:
        dilate_mask=args.dilate_mask

        local_mask = process_mask(args.mask_input[0], args.height, args.width, dilate=dilate_mask,
                                  dilation_kernel_size=(5, 5), fill_holes=args.fill_hole_mask,
                                  closing_kernel_size=(1, 1))
        mask_path_list.append(args.mask_input[0])
        local_mask_addition = []
        kkk=1
        for maskp in args.mask_input[1:]:
            dilate_mask = args.dilate_mask
            local_mask_addition.append(
                process_mask(maskp, args.height, args.width, dilate=dilate_mask, dilation_kernel_size=(5, 5),
                             fill_holes=args.fill_hole_mask, closing_kernel_size=(1, 1)))
            mask_path_list.append(maskp)
            kkk+=1
    ## if point_flag==1, let the masks be the points
    if args.mask_input != None and point_flag==1:
        local_mask =parsed_inputs[0]
        local_mask_addition=parsed_inputs[1:]


    # Create the masks online if not given
    if args.mask_input == None or args.self_infer_point==True:

        ## if args.self_infer_point==True, use InternVL3 to infer point hints
        ## if args.self_infer_point==False, ask users to select mode of interactively creating masks
        '''
        Modes of interactively creating masks
        Press '1' - interactively draw an oval mask, usually used in addition: use an oval binary mask to indicate the location to add objects;
        Press '2' - interactively use SAM to segment object mask, usually used in replacement or removal: segment to object to be replaced or to be removed;
        Press '3' - interactively select a point in the image to indicate the region around the point to edit, usually used in object addition;
        Press '4' - Use  InternVL3 to infer point hints for object additions.
        '''
        if args.self_infer_point==False:
            from helper.segment_anything_gui import run_gui
            from helper.draw_oval_mask import draw_oval_mask
            print('\n')
            print(
                'No mask path provided. Now we will create the masks online. For this, if you are running on an online server, you need to enable Forward X11 Connections \n')


            while True:
                resp = input(
                    "Press '1' for Interactively Drawing the Mask, Press '2' for Applying SAM to get object masks, Press '3' for Picking a point as an approximate location to add object, Press '4' for using VLM to automatically infer location to edit:  ")
                if resp == '1' or resp == '2' or resp=='3' or resp=='4':
                    break
                else:
                    print("Invalid input. Please press '1','2','3' or '4'.")
        elif args.self_infer_point==True:
            resp='4'
        ## Use  InternVL3 to infer point hints for object additions.
        if resp == '4':
            point_flag=1
            import math
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            from transformers import AutoModel, AutoTokenizer
            from helper.internvl import load_image_internvl

            intern_path = args.internvl_checkpoint_path
            intern_model = AutoModel.from_pretrained(
                intern_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True).eval().cuda()
            tokenizer = AutoTokenizer.from_pretrained(intern_path, trust_remote_code=True, use_fast=False)

            pixel_values = load_image_internvl(args.image_path, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=1024, do_sample=False, temperature=0.0)

            if len(args.prompt_local) == 1:
                # Run inference
                local_prompt=args.prompt_local[0]
                promptt="Let's say I want to add a new subject to an image:"+str(local_prompt) +" Could you suggest a bounding box for this subject? You can first consider the bounding box of the object the subject is near, supporting, or otherwise in relation to. Then, analyze the position of the added subject relative to the object (left, right, above, or below? specifc this) and assign a bounding box to the added subject. Note that the added subject should not overlap with existing subjects in the image generally and size of the box should allow the subject to fit in. Please give me the bounding box in format [[x1,y1,x2,y2]]. Norm the value between 0 and 1000"
                question ='<image>\n'+str(promptt)
                response = intern_model.chat(tokenizer, pixel_values, question, generation_config)
                print('The inference result of the point hints \n')
                print(f'Assistant: {response}')
                del intern_model,tokenizer
                import re
                #Read the point hint from the output text
                boxes = re.findall(r"\[\[\d+,\d+,\d+,\d+\]\]",response)
                if boxes==[]:
                    boxes = re.findall(r"\[\[\d+, \d+, \d+, \d+\]\]",response)
                last_box = eval(boxes[-1]) if boxes else None
                bounding_box=last_box[0]
                x_min, y_min, x_max, y_max = bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3]
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                x_center_norm = x_center / 1000
                y_center_norm = y_center / 1000
                local_mask=(x_center_norm,y_center_norm)
                local_mask_addition = []
                print('The inferred point hint for adding the object:'+str(args.prompt_local[0])+' is shown with the yellow star.')
                visualize_with_star(args.image_path, local_mask[0],local_mask[1])
            if len(args.prompt_local) > 1:
                # Start building the grounding prompt
                local_mask_addition = []
                grounding_prompt = "Let's say I want to add "+str(len(args.prompt_local))+" new subjects to an image. "

                # Loop through the local prompts and dynamically add them
                for idd, prompt in enumerate(args.prompt_local):
                    subject = f"Subject {chr(65 + idd)}"  # Converts 0 -> A, 1 -> B, 2 -> C, etc.
                    grounding_prompt += f"{subject}: {str(prompt)} "

                # Add the final instructions for the bounding box
                grounding_prompt += "Could you suggest a bounding box for these subjects? You can first consider the bounding box of the object the subject is near, supporting, or otherwise in relation to. Then, analyze the position of the added subject relative to the object (left, right, above, or below? specify this) and assign a bounding box to the added subject. Note that the added subject should not overlap with existing subjects in the image generally and the size of the box should allow the subject to fit in. Please give me the bounding box in format "

                # Add bounding box format for all subjects
                bounding_boxes_format = ", ".join([f"Subject {chr(65 + ii)}: [[x1,y1,x2,y2]]" for ii in range(len(args.prompt_local))])
                grounding_prompt += bounding_boxes_format
                promptt=grounding_prompt

                question ='<image>\n'+str(promptt)  #'<image>\nPlease describe the image in detail.'
                response = intern_model.chat(tokenizer, pixel_values, question, generation_config)
                print('The inference result of the point hints \n')
                print(f'Assistant: {response}')
                del intern_model,tokenizer
                import re
                #Read the point hints from the output text
                bounding_boxes = re.findall(r"\[\[\d+, \d+, \d+, \d+\]\]",response)
                for kk in range(len(args.prompt_local)):
                    last_box = eval(bounding_boxes[kk])
                    subject_bounding_box=last_box[0]
                    x_min, y_min, x_max, y_max = subject_bounding_box[0],subject_bounding_box[1],subject_bounding_box[2],subject_bounding_box[3]
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    x_center_norm = x_center / 1000
                    y_center_norm = y_center / 1000
                    if (y_center_norm+0.15)<=0.8:
                        y_center_norm+=0.15
                    elif (y_center_norm+0.1)<=0.8:
                        y_center_norm+=0.1
                    if kk==0:
                        local_mask=(x_center_norm,y_center_norm)
                    else:
                        local_mask_addition.append((x_center_norm,y_center_norm))
                    print('The inferred point hint for adding the object:'+str(args.prompt_local[kk])+' is shown with the yellow star.')
                    visualize_with_star(args.image_path, x_center_norm,y_center_norm)

        else:
            print('Create mask for the first region to edit: ' + str(args.prompt_local[0]))
            ##interactively select a point in the image to indicate the region around the point to edit, usually used in object addition;
            if resp == '3':
                from helper.select_point import select_single_point
                print('\n')
                print("Left mouse click to get a point, and press 'd' to confirm the point and exit")
                print('\n')
                local_mask=select_single_point(args.image_path,args.width,args.height)
                point_flag=1
            ##interactively use SAM to segment object mask, usually used in replacement or removal: segment to object to be replaced or to be removed;
            if resp == '2':
                print('Launching the GUI')
                mask_path0 = run_gui(img_input_filepath=args.image_path, output_dir="./mask_temp/", new_width=args.width,
                                     new_height=args.height, sam_checkpoint=args.sam_checkpoint)
                dilate_mask = args.dilate_mask
                local_mask = process_mask(mask_path0, args.height, args.width, dilate=dilate_mask,
                                          dilation_kernel_size=(5, 5), fill_holes=args.fill_hole_mask,
                                          closing_kernel_size=(1, 1))  #
                mask_path_list.append(mask_path0)
            ##interactively draw an oval mask, usually used in addition: use an oval binary mask to indicate the location to add objects;
            if resp == '1':
                print('\n')
                print("press ENTER to confirm a mask after drawing, and after that press 'd' to exit")
                print('\n')
                dilate_mask = args.dilate_mask
                mask_path0 = draw_oval_mask(args.image_path, save_dir="./mask_temp/", new_width=args.width,
                                            new_height=args.height)
                local_mask = process_mask(mask_path0, args.height, args.width, dilate=dilate_mask,
                                          dilation_kernel_size=(5, 5), fill_holes=args.fill_hole_mask,
                                          closing_kernel_size=(1, 1))
                mask_path_list.append(mask_path0)
            print('\n')
            local_mask_addition = []
            if len(args.prompt_local) > 1:
                kkk=1
                for pp in args.prompt_local[1:]:
                    print('\n')
                    print('Create mask for the next region to edit: ' + str(pp))
                    while True:
                        resp = input(
                            "Press '1' for Interactively Drawing the Mask, Press '2' for Applying SAM to get object masks, Press '3' for Picking a point as an approximate location to add object:")
                        if resp == '1' or resp == '2' or resp=='3':
                            break
                        else:
                            print("Invalid input. Please press '1','2','3'.")
                    ## interactively select a point in the image to indicate the region around the point to edit, usually used in object addition;
                    if resp == '3':
                        print('\n')
                        print("Left mouse click to get a point, and press 'd' to confirm the point and exit")
                        print('\n')
                        local_mask_addition.append(select_single_point(args.image_path,args.width,args.height))
                        point_flag=1
                    ## interactively use SAM to segment object mask, usually used in replacement or removal: segment to object to be replaced or to be removed;
                    if resp == '2':
                        dilate_mask = args.dilate_mask
                        print('Launching the GUI')
                        mask_path1 = run_gui(img_input_filepath=args.image_path, output_dir="./mask_temp/",
                                             new_width=args.width, new_height=args.height,
                                             sam_checkpoint=args.sam_checkpoint)
                        local_mask_addition.append(
                            process_mask(mask_path1, args.height, args.width, dilate=dilate_mask,
                                         dilation_kernel_size=(5, 5), fill_holes=args.fill_hole_mask,
                                         closing_kernel_size=(1, 1)))
                        mask_path_list.append(mask_path1)
                    ## interactively draw an oval mask, usually used in addition: use an oval binary mask to indicate the location to add objects;
                    if resp == '1':
                        print('\n')
                        print("press ENTER to confirm a mask after drawing, and after that press 'd' to exit")
                        print('\n')
                        dilate_mask = args.dilate_mask
                        mask_path1 = draw_oval_mask(args.image_path, save_dir="./mask_temp/", new_width=args.width,
                                                    new_height=args.height)
                        mask_path_list.append(mask_path1)
                        local_mask_addition.append(
                            process_mask(mask_path1, args.height, args.width, dilate=dilate_mask,
                                         dilation_kernel_size=(5, 5), fill_holes=args.fill_hole_mask,
                                         closing_kernel_size=(1, 1)))
                    print('\n')
                    kkk+=1

    ##==============END of Prepare Masks======================================================================================


    # ------------------------------------------------------------------------------------------------------------------------
    '''
    When make refinement after state 1 is enabled OR
    multiple runs of edits are enable (either re-generate based on current prompts&masks but different seeds
    or edit on top of current edited results with new prompts&masks):
    cache the inversion result at the args.inversion_save_path;
    cache the previous generation noisy latents at args.generate_save_path.
    '''
    if args.refine_mask == True or args.multi_run==True:
        if args.generate_save_path == None:
            if not os.path.exists('./info_dict_folder'): os.mkdir('./info_dict_folder')
            args.generate_save_path = './info_dict_folder/prev_gen.npy'
        if args.inversion_save_path == None:
            if not os.path.exists('./info_dict_folder'): os.mkdir('./info_dict_folder')
            args.inversion_save_path = './info_dict_folder/running.npy'

    ## generation_flag is to decide whether to continue the CannyEdit or exit
    ## regeneration_flag is to decide whether the current generation is doing the re-generation (either re-generate based on current prompts&masks or edit on top of current edited results with new prompts&masks )
    generation_flag=True
    regeneration_flag=False
    ## if args.generate_next_round_flag==True: the edit is to do new edit on top of current edited results with new prompts&masks
    if args.generate_next_round_flag==True:regeneration_flag=True
    ### ----------------------------------------------------------------------------------------------------------------------
    ### Start to Run CannyEdit
    while generation_flag == True:
        '''
        The code support three modes:
        'stage_removal': object removal mode;
        'stage_generate': object addition, replacement mode with binary masks provided;
        'stage_generate_point': object addition mode with point hints provided.
        '''
        ##==============First stage of CannyEdit==============================================================================
        if (removal_flag == True) or (removal_flag==False and point_flag==False):
            ##if removal_flag == True: removal mode
            ##if removal_flag==False and point_flag==False: cannyedit with binary masks provided
            if removal_flag == True:stage1='stage_removal'
            if removal_flag==False and point_flag==False:stage1 ='stage_generate'
            if regeneration_flag: stage1=stage1 + '_regen'

            print('Running CannyEdit')
            #### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Stage 1: Generation @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            result = xflux_pipeline(
                prompt_source=args.prompt_source,
                prompt_local1=args.prompt_local[0],
                prompt_target=args.prompt_target,
                prompt_local_addition=args.prompt_local[1:],
                controlnet_image=image,
                local_mask=local_mask,
                local_mask_addition=local_mask_addition,
                width=args.width,
                height=args.height,
                guidance=args.guidance,
                num_steps=args.num_steps,
                seed=args.seed,
                true_gs=args.true_gs,
                control_weight=args.control_weight,
                control_weight2=args.control_weight2,
                neg_prompt=args.neg_prompt,
                neg_prompt2=args.neg_prompt2,
                timestep_to_start_cfg=args.timestep_to_start_cfg,
                stage=stage1,
                generate_save_path=args.generate_save_path,
                inversion_save_path=args.inversion_save_path)

            ## Save the edited image
            if args.save_folder.endswith(".png"):
                result_save_path = args.save_folder
            else:
                if not os.path.exists(args.save_folder):
                    os.mkdir(args.save_folder)
                ind = len(os.listdir(args.save_folder))
                result_save_path = os.path.join(args.save_folder, f"result_{ind}.png")
            result.save(result_save_path)

            if removal_flag == True and args.refine_mask == True:
                print('No mask refinement function is provided for the removal task.')


        if point_flag==True:
            #### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Stage 1: Generation @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            ## for the 'stage_generate_point': object addition mode with point hints provided, after the first stage of generation, the extraction of masks for objects to add is required.
            stage1 ='stage_generate_point'
            if regeneration_flag: stage1=stage1 + '_regen'
            print('Running CannyEdit')
            result = xflux_pipeline(
                prompt_source=args.prompt_source,
                prompt_local1=args.prompt_local[0],
                prompt_target=args.prompt_target,
                prompt_local_addition=args.prompt_local[1:],
                controlnet_image=image,
                local_mask=local_mask,
                local_mask_addition=local_mask_addition,
                width=args.width,
                height=args.height,
                guidance=args.guidance,
                num_steps=args.num_steps,
                seed=args.seed,
                true_gs=args.true_gs,
                control_weight=args.control_weight,
                control_weight2=args.control_weight2,
                neg_prompt=args.neg_prompt,
                neg_prompt2=args.neg_prompt2,
                timestep_to_start_cfg=args.timestep_to_start_cfg,
                stage=stage1,
                generate_save_path=args.generate_save_path,
                inversion_save_path=args.inversion_save_path,
                norm_softmask=args.norm_softmask)

            ## Save the edited image
            if args.save_folder.endswith(".png"):
                result_save_path = args.save_folder
            else:
                if not os.path.exists(args.save_folder):
                    os.mkdir(args.save_folder)
                ind = len(os.listdir(args.save_folder))
                result_save_path = os.path.join(args.save_folder, f"result_{ind}.png")
            result.save(result_save_path)

            ####Start the the extraction of masks for objects to add
            handson_flag_list=[]
            point_hint_flag=True
            threshold_soft1=0.35
            # Initialize models
            from helper.auto_get_masks_point import extract_mask_fromsoft,auto_get_mask
            from helper.lang_sam import LangSAM
            from helper.sam_utils import SAM
            from helper.auto_get_masks_point import auto_get_mask
            from transformers import AutoModelForCausalLM, AutoTokenizer
            args.save_path = result_save_path
            ##apply Language SAM for extracting masks
            ls_model = LangSAM(sam_type="sam2.1_hiera_large", sam_ckpt_path=args.sam_checkpoint,
                               gdino_ckpt_path=args.gdino_checkpoint)
            sam = SAM()
            sam.build_model("sam2.1_hiera_large", args.sam_checkpoint, device='cuda')
            predictor = sam.predictor
            ##apply Qwen LLM for obtaining the label texts to fit in Language SAM for extracting masks
            model_name = args.qwen_llm_checkpoint_path
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            qwen_model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map="auto")

            final_mask_list=[]
            non_mask_found_str=''
            ## if the automatical mask extraction is failed, return a handson_flag=True to indicate that users need to do the mask extraction manually.
            new_local_mask, handson_flag,final_mask = extract_mask_fromsoft(args=args,center=local_mask,height=args.height,width=args.width, prompt_local=args.prompt_local[0],result_save_path=result_save_path,\
                                                                 ls_model=ls_model,predictor=predictor,qwen_model=qwen_model,qwen_processor=tokenizer,threshold_soft=threshold_soft1, radius=250,\
                                                                 dilate_mask=args.dilate_mask,fill_hole_mask=args.fill_hole_mask,max_points=5,output_dir="./mask_temp/",final_mask_list=final_mask_list,mask_save_name=image_name+'_0')
            final_mask_list.append(final_mask)
            if handson_flag==True:
                non_mask_found_str+=args.prompt_local[0]+';'
            local_mask1=new_local_mask
            local_mask_addition1=copy.deepcopy(local_mask_addition)
            handson_flag_list.append(handson_flag)
            if len(args.prompt_local) > 1:
                kk1=0
                for pp in args.prompt_local[1:]:
                    new_local_mask, handson_flag,final_mask = extract_mask_fromsoft(args=args, center=local_mask_addition1[kk1],
                                                                         height=args.height, width=args.width,prompt_local=args.prompt_local[kk1+1],
                                                                         result_save_path=result_save_path,ls_model=ls_model, predictor=predictor,
                                                                         qwen_model=qwen_model,qwen_processor=tokenizer,
                                                                         threshold_soft=threshold_soft1, radius=250,
                                                                         dilate_mask=args.dilate_mask,fill_hole_mask=args.fill_hole_mask, max_points=5, output_dir="./mask_temp/",
                                                                         final_mask_list=final_mask_list,mask_save_name=image_name+'_'+str(kk1+1))
                    local_mask_addition1[kk1]=new_local_mask
                    final_mask_list.append(final_mask)
                    handson_flag_list.append(handson_flag)
                    if handson_flag==True:
                        non_mask_found_str+=args.prompt_local[kk1+1]+';'
                    kk1+=1
            del predictor
            del ls_model
            del qwen_model
            del tokenizer


            if True in handson_flag_list:
                print('No mask found for '+str(non_mask_found_str)+' The editing result at stage 1 is shown')
                img = mpimg.imread(result_save_path)
                plt.figure(figsize=(10, 10))
                plt.imshow(img)
                plt.axis('off')  # Turn off the axes
                plt.show()
                # Load and display the image
                resp1 = input(
                    "Do you want to re-generate the edit now or manually select the mask? Press 'r' to re-do the generation now, and press anything else to manually select the mask: ").strip().lower()
                if resp1=='r':point_regenerate=True

        ##==============End of First stage of CannyEdit=======================================================================

        ##==============Second stage of CannyEdit (Optionally, except it is a must for editing with point hints)===========================
        #### @@@@@@@@@@@@@@@ Stage 2: Generated with refined masks and previous generation results @@@@@@@@@@@@@@@
        if args.refine_mask == True and removal_flag==True:
            print('Mask refinement is disabled for the removal.')
        if args.refine_mask == True and removal_flag==False:
            #----- Automatically generate the refined masks for the case of binary masks provided ----------------
            if args.auto_mask_refine==True:
                print('Using Language SAM to automatically gather masks for mask refinement')
                args.save_path=result_save_path
                from helper.lang_sam import LangSAM
                from helper.auto_get_masks import auto_get_mask
                ls_model = LangSAM(sam_type="sam2.1_hiera_large", sam_ckpt_path=args.sam_checkpoint,
                                   gdino_ckpt_path=args.gdino_checkpoint)

                from transformers import AutoModelForCausalLM, AutoTokenizer
                model_name = args.qwen_llm_checkpoint_path
                # load the tokenizer and the model
                qwen_processor = AutoTokenizer.from_pretrained(model_name)
                qwen_model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map="auto")

                promptt = 'Given this prompt describe a local region of an image:' + str(
                    args.prompt_local[0]) +"Based on this, give a noun in 3 words for the main subject in this region (if the main subject is a human, specific character if needed, still within 3 words). If the gender and age is in the given prompt, mention them. Give the description directly without any other contents"

                messages = [
                    {"role": "user", "content": promptt}]
                text = qwen_processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = qwen_processor([text], return_tensors="pt").to(qwen_model.device)

                # conduct text completion
                generated_ids = qwen_model.generate(
                    **model_inputs,
                    max_new_tokens=16384
                )
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                content = qwen_processor.decode(output_ids, skip_special_tokens=True)
                label_prompt = content.strip().lower()
                print(label_prompt)

                simple_mask=local_mask
                try:
                    mask_path0=auto_get_mask(ls_model, label_prompt,simple_mask,args,output_dir="./mask_temp/")
                    local_mask1 = process_mask(mask_path0, args.height, args.width, dilate=args.dilate_mask,
                                              dilation_kernel_size=(5, 5), fill_holes=args.fill_hole_mask,
                                              closing_kernel_size=(1, 1))
                except Exception as e:
                    local_mask1=local_mask
                    print('no refined mask found for '+str(label_prompt)+', use previous mask')
                local_mask_addition1=copy.deepcopy(local_mask_addition)


                if len(args.prompt_local) > 1:
                    kkk = 0
                    for pp in args.prompt_local[1:]:
                        promptt = 'Given this prompt describe a local region of an image:' + str(
                           pp) +   "Based on this, give a noun in 3 words for the main subject in this region (if the main subject is a human, specific character if needed, still within 3 words). If the gender and age is in the given prompt, mention them. Give the description directly without any other contents"
                        messages = [
                            {"role": "user", "content": promptt}
                        ]
                        text = qwen_processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        model_inputs = qwen_processor([text], return_tensors="pt").to(qwen_model.device)

                        # conduct text completion
                        generated_ids = qwen_model.generate(
                            **model_inputs,
                            max_new_tokens=16384
                        )
                        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

                        content = qwen_processor.decode(output_ids, skip_special_tokens=True)
                        label_prompt = content.strip().lower()
                        print(label_prompt)

                        simple_mask=local_mask_addition1[kkk]
                        try:
                            mask_path1 = auto_get_mask(ls_model, label_prompt,simple_mask,args,output_dir="./mask_temp/")
                            local_mask_addition1[kkk]=process_mask(mask_path1, args.height, args.width, dilate=args.dilate_mask,
                                             dilation_kernel_size=(5, 5), fill_holes=args.fill_hole_mask,
                                             closing_kernel_size=(1, 1))
                        except:
                            print('no refined mask found for ' + str(pp) + ', use previous mask')
                        kkk+=1

                del ls_model
                del qwen_model
                del qwen_processor


            #------ User hand-draft the refined masks [if args.auto_mask_refine == False or automatically extracting masks is failed]   ----------------
            if (point_flag==0 and args.auto_mask_refine == False) or (point_flag==1 and True in handson_flag_list and point_regenerate==False):


                if point_flag==0:
                    print('\n')
                    print('First run of generation is done, please check the generated result and decide whether to refine the generation. \n')
                    from helper.segment_anything_gui import run_gui
                    print('Refine mask for the first edit region: ' + str(args.prompt_local[0]))
                    mask_path0 = run_gui(img_input_filepath=result_save_path, output_dir="./mask_temp/", new_width=args.width,
                                         new_height=args.height, sam_checkpoint=args.sam_checkpoint)
                    local_mask1 = process_mask(mask_path0, args.height, args.width, dilate=args.dilate_mask,
                                              dilation_kernel_size=(5, 5), fill_holes=args.fill_hole_mask,
                                              closing_kernel_size=(1, 1))  #
                    local_mask_addition1=[]
                    if len(args.prompt_local) > 1:
                        for pp in args.prompt_local[1:]:
                            print('\n')
                            print('Refine mask for the next edit region: ' + str(pp))
                            mask_path1 = run_gui(img_input_filepath=result_save_path, output_dir="./mask_temp/",
                                                 new_width=args.width, new_height=args.height,
                                                 sam_checkpoint=args.sam_checkpoint)
                            local_mask_addition1.append(
                                process_mask(mask_path1, args.height, args.width, dilate=args.dilate_mask,
                                             dilation_kernel_size=(5, 5), fill_holes=args.fill_hole_mask,
                                             closing_kernel_size=(1, 1)))
                elif point_flag==1:
                    print('\n')
                    #print('First run of generation is done, please check the generated result and decide whether to refine the generation. \n')
                    print(
                        'Start to manually refine mask. \n')
                    from helper.segment_anything_gui import run_gui

                    if handson_flag_list[0]==True:
                        print('Manually refine mask for the first edit region: ' + str(args.prompt_local[0]))
                        mask_path0 = run_gui(img_input_filepath=result_save_path, output_dir="./mask_temp/", new_width=args.width,
                                             new_height=args.height, sam_checkpoint=args.sam_checkpoint)
                        local_mask1 = process_mask(mask_path0, args.height, args.width, dilate=args.dilate_mask,
                                                  dilation_kernel_size=(5, 5), fill_holes=args.fill_hole_mask,
                                                  closing_kernel_size=(1, 1))  #
                    local_mask_addition1=copy.deepcopy(local_mask_addition)
                    kkk1=1
                    if len(args.prompt_local) > 1:
                        for pp in args.prompt_local[1:]:
                            if handson_flag_list[kkk1]==True:
                                print('\n')
                                print('Manually refine mask for the edit region: ' + str(pp))
                                mask_path1 = run_gui(img_input_filepath=result_save_path, output_dir="./mask_temp/",
                                                     new_width=args.width, new_height=args.height,
                                                     sam_checkpoint=args.sam_checkpoint)
                                local_mask_addition1[kkk1-1]=\
                                    process_mask(mask_path1, args.height, args.width, dilate=args.dilate_mask,
                                                 dilation_kernel_size=(5, 5), fill_holes=args.fill_hole_mask,
                                                 closing_kernel_size=(1, 1))
                            kkk1+=1

            # ------ Running CannyEdit with refined masks  ----------------
            print('Running CannyEdit with refined masks')
            if point_flag==0:
                result = xflux_pipeline(
                    prompt_source=args.prompt_source,
                    prompt_local1=args.prompt_local[0],
                    prompt_target=args.prompt_target,
                    prompt_local_addition=args.prompt_local[1:],
                    controlnet_image=image,
                    local_mask=local_mask1,
                    local_mask_addition=local_mask_addition1,
                    width=args.width,
                    height=args.height,
                    guidance=args.guidance,
                    num_steps=args.num_steps,
                    seed=args.seed,
                    true_gs=args.true_gs,
                    control_weight=args.control_weight,
                    control_weight2=args.control_weight2,
                    neg_prompt=args.neg_prompt,
                    timestep_to_start_cfg=args.timestep_to_start_cfg,
                    stage='stage_refine',
                    generate_save_path=args.generate_save_path,
                    inversion_save_path=args.inversion_save_path
                )
                ## Save the refined edited image
                ind = len(os.listdir(args.save_folder))
                result_save_path = os.path.join(args.save_folder, f"result_{ind-1}_refine.png")
                result.save(result_save_path)
            elif point_flag==1 and point_regenerate==False:
                result = xflux_pipeline(
                    prompt_source=args.prompt_source,
                    prompt_local1=args.prompt_local[0],
                    prompt_target=args.prompt_target,
                    prompt_local_addition=args.prompt_local[1:],
                    controlnet_image=image,
                    local_mask=local_mask1,
                    local_mask_addition=local_mask_addition1,
                    width=args.width,
                    height=args.height,
                    guidance=args.guidance,
                    num_steps=args.num_steps,
                    seed=args.seed,
                    true_gs=args.true_gs,
                    control_weight=args.control_weight,
                    control_weight2=args.control_weight2,
                    neg_prompt=args.neg_prompt,
                    timestep_to_start_cfg=args.timestep_to_start_cfg,
                    stage='stage_refine_point',
                    generate_save_path=args.generate_save_path,
                    inversion_save_path=args.inversion_save_path,
                    norm_softmask=args.norm_softmask
                )
                ## Save the refined edited image
                ind = len(os.listdir(args.save_folder))
                result_save_path = os.path.join(args.save_folder, f"result_{ind-1}_refine.png")
                result.save(result_save_path)

        ##==============End of second stage of CannyEdit======================================================================

        ##Decide to do the re-generation or not
        ##  (either re-generate based on current prompts&masks but different seeds or edit on top of current edited results with new prompts&masks )
        # if args.multi_run == False (default): stop the generation
        if args.multi_run == False:
            generation_flag=False
        # if point_regenerate==True, redo the generation with point hints
        if  point_flag==1 and point_regenerate==True:
            args.seed=random.randint(0, 9999999)
            generation_flag = True
            regeneration_flag=True
            point_regenerate=False
            print('Start Regeneration')
        if args.multi_run == True and point_regenerate==False:
            args.seed=random.randint(0, 9999999)
            ## show generation result at current run
            print('The editing result at current generation pass is shown')
            img = mpimg.imread(result_save_path)
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.axis('off')  # Turn off the axes
            plt.show()
            # Load and display the image
            resp = input(
                "Press 'y' to re-generate the edit at current run; Press 'n' to generate next round of edit based on current; Press 'g' to give up current edit and continue the edit based on previous edits; and Press anything else to exist: ").strip().lower()
            if resp == "y":
                generation_flag = True
                regeneration_flag=True
                print('Start Regeneration')
            elif resp == "n":
                args.mask_input = None
                args.prompt_local = None
                args.prompt_source = None
                args.prompt_target = None
                args.image_path=result_save_path
                args.generate_next_round_flag=True
                args.previous_generate_latent=np.load(args.generate_save_path, allow_pickle=True).item()
                np.save(args.inversion_save_path, args.previous_generate_latent)
                main(args)
            elif resp == "g":
                args.mask_input = None
                args.prompt_local = None
                args.prompt_source = None
                args.prompt_target = None
                args.image_path=result_save_path
                args.generate_next_round_flag=True
                args.previous_generate_latent=args.previous_generate_latent
                np.save(args.inversion_save_path, args.previous_generate_latent)
                main(args)
            else:
                generation_flag = False
                regeneration_flag = False
                print('\n')
                print('Exiting Current Generation')
                if args.inversion_save_path!=None and os.path.exists(args.inversion_save_path):
                    os.remove(args.inversion_save_path)
                if args.generate_save_path!=None and os.path.exists(args.generate_save_path):
                    os.remove(args.generate_save_path)
                sys.exit(1)


    ### remove all cached files
    if args.multi_run==False or (args.multi_run==True and resp!='n' and resp!='g'):
        if args.inversion_save_path!=None and os.path.exists(args.inversion_save_path):
            os.remove(args.inversion_save_path)
    if args.generate_save_path!=None and os.path.exists(args.generate_save_path):
        os.remove(args.generate_save_path)


# ============================================================================================================================

if __name__ == "__main__":
    args = create_argparser().parse_args()
    xflux_pipeline = XFluxPipeline("flux-dev", "cuda", False)
    xflux_pipeline.set_controlnet("canny", None, "XLabs-AI/flux-controlnet-canny-v3","flux-canny-controlnet-v3.safetensors")
    args.pipe = xflux_pipeline

    args.qwen_checkpoint_path =  "/project/imgtextmod/model_checkpoints/Qwen2.5-VL-7B-Instruct"
    args.qwen_llm_checkpoint_path =  "/project/imgtextmod/model_checkpoints/Qwen3-4B-Instruct-2507"
    args.internvl_checkpoint_path= "/project/imgtextmod/model_checkpoints/InternVL3-14B"
    args.sam_checkpoint = "/project/imgtextmod/model_checkpoints/sam2.1_hiera_large.pt"
    args.gdino_checkpoint="/project/imgtextmod/model_checkpoints/grounding-dino-base"
    main(args)
