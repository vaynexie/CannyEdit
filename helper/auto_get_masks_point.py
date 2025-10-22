
## code for refining the mask from the point-inferred soft mask
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import cv2
import copy
import matplotlib
from helper.sam_utils import *

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



def create_soft_mask(mask, center, height, width):
    """
    Create a soft mask where values increase from 0 at the center to 1 at the farthest point.

    Args:
        mask (torch.Tensor): A 2D binary mask (height x width).
        center (tuple): The center point (x, y) as normalized coordinates (between 0 and 1).
        height (int): Height of the mask.
        width (int): Width of the mask.

    Returns:
        torch.Tensor: A 2D soft mask with values ranging from 0 to 1.
    """
    mask = mask.view(height, width)

    # Scale normalized center to pixel coordinates
    center_x = center[0] * width
    center_y = center[1] * height

    # Create coordinate grids
    y_coords = torch.arange(height, dtype=torch.float32).view(-1, 1).expand(height, width)
    x_coords = torch.arange(width, dtype=torch.float32).view(1, -1).expand(height, width)

    # Compute squared Euclidean distances from the center
    distances = (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2

    # Normalize distances to [0, 1]
    max_distance = distances.max()
    soft_mask = distances / max_distance

    return soft_mask


def extract_mask_fromsoft(
    args,
    center,
    height: int,
    width: int,
    prompt_local: str,
    result_save_path: str,
    ls_model,
    predictor,
    qwen_model,
    qwen_processor,
    threshold_soft: float = 0.3,
    radius: int = 120,
    dilate_mask: bool = False,
    fill_hole_mask: bool = True,
    closing_kernel_size: tuple = (1, 1),
    dilation_kernel_size: tuple = (2, 2),
    max_points: int = 5,
    output_dir="./mask_temp/",
    device: str = None,
    final_mask_list=[],
    mask_save_name=None

):
    """
    Returns:
        (new_local_mask: torch.Tensor or None, handson_flag: bool)

    If handson_flag == True, new_local_mask is None.
    """



    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Downsampled grid sizes
    h16, w16 = height // 16, width // 16

    # 1) Create soft mask around center


    soft_mask = create_soft_mask(torch.ones((h16,w16)), center, h16, w16).to(device)

    # 2) Threshold soft mask into 0/100 per your original
    soft_mask = torch.where(soft_mask < threshold_soft, torch.tensor(0, device=device), torch.tensor(100, device=device))




    print("Using Language SAM to automatically gather masks for mask refinement")



    promptt = (
            "Given this prompt describe a local region of an image: "
            + str(prompt_local)
            #+"Based on this, give a noun in 2 words for the main subject in this region (if the main subject is a human, specific character,gender if needed, still within 2 words). Give the description directly without any other contents"
            + "Based on this, give a noun in 3 words for the main subject in this region (if the main subject is a human, specific character if needed, still within 3 words). If the gender and age is in the given prompt, mention them. Give the description directly without any other contents"

    )

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
    print(f"Qwen label: {label_prompt}")



    handson_flag = False

    try:
    #if 1==1:
        # 5) Use LangSAM to get a coarse mask conditioned on label and soft prior
        simple_mask = soft_mask
        mask_path0,final_mask = auto_get_mask(ls_model, label_prompt, simple_mask,args,
                                   output_dir=output_dir,final_mask_list=final_mask_list,mask_save_name=mask_save_name)

        # 6) Process the coarse mask to base resolution
        local_mask_processed = process_mask(
            mask_path0, height, width,
            dilate=False,
            dilation_kernel_size=dilation_kernel_size,
            fill_holes=False,
            closing_kernel_size=closing_kernel_size
        )

        # 7) Build attention-like map as inverse mask, upsample to HxW
        local_mask11 = 1 - copy.deepcopy(local_mask_processed)
        concatenated_tensors_mean = local_mask11
        tensor_reshaped = concatenated_tensors_mean.view(h16, w16).unsqueeze(0).unsqueeze(0)
        tensor_upsampled = F.interpolate(tensor_reshaped, size=(height, width), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)


        radius_min=int(60*(np.max([height,width])/512))
        radius_use = np.max([radius_min,int(local_mask11.sum() / (local_mask11.shape[0] * local_mask11.shape[1]) * radius)*3])
        print(f"Radius use: {radius_use}")
        # 8) Greedy point selection with radius suppression
        points = []
        top_values = []
        tensor_copy = tensor_upsampled.clone()
        pmax = torch.max(tensor_copy)
        threshold = 0.65 * pmax

        while len(points) < max_points:
            flat_idx = tensor_copy.argmax()
            max_pos = torch.unravel_index(flat_idx, tensor_copy.shape)
            max_poss = [max_pos[0].item(), max_pos[1].item()]
            max_val = tensor_copy[max_pos[0].item(), max_pos[1].item()]
            if max_val < threshold:
                break
            points.append(max_poss)
            top_values.append(max_val)

            row, col = max_poss
            r0 = max(0, row - radius_use)
            r1 = min(tensor_copy.size(0), row + radius_use + 1)
            c0 = max(0, col - radius_use)
            c1 = min(tensor_copy.size(1), col + radius_use + 1)
            tensor_copy[r0:r1, c0:c1] = 0

        # 9) Run SAM on selected points to refine the mask


        image_path = result_save_path
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        predictor.set_image(image_np)

        input_point = np.array([[p[1], p[0]] for p in points], dtype=np.int32) if len(points) > 0 else np.empty((0, 2), dtype=np.int32)
        input_label = np.array([1] * len(input_point), dtype=np.int32) if len(points) > 0 else np.empty((0,), dtype=np.int32)

        masks, scores, logits = predictor.predict(
            point_coords=input_point if len(points) > 0 else None,
            point_labels=input_label if len(points) > 0 else None,
            multimask_output=True,
        )

        sorted_ind = np.argsort(scores)[::-1][:1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]


        # 10) Save final mask to disk and re-process to align with training mask pipeline
        final_mask = masks[0]

        # Desired new dimensions
        new_width, new_height = height, width

        # Resize the mask using nearest-neighbor interpolation
        resized_mask = cv2.resize(final_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # Ensure the resized mask is still binary (0 and 1)
        final_mask = (resized_mask > 0).astype(np.uint8)


        os.makedirs(output_dir, exist_ok=True)
        ind = len(os.listdir(output_dir))
        if mask_save_name==None:
            mask_file = f"mask_{ind}.png"
        else:
            mask_file = 'samfinal_'+mask_save_name+'.png'
        mask_path = os.path.join(output_dir, mask_file)

        mask_output = np.where(final_mask != 0, 255, 0).astype(np.uint8)
        cv2.imwrite(mask_path, mask_output)
        #dilation_kernel_size = (dilation_kernel_size[0] * int(height / 512), dilation_kernel_size[1] * int(width / 512))

        new_local_mask = process_mask(
            mask_path, height, width,
            dilate=False,
            dilation_kernel_size=dilation_kernel_size,
            fill_holes=fill_hole_mask,
            closing_kernel_size=closing_kernel_size
        )



        return new_local_mask, False,final_mask

    except Exception as e:
    #else:
        print(f"Fail to automatically find mask for '{label_prompt}', please manually select the mask")
        handson_flag = True

        return None, handson_flag,None




def auto_get_mask(ls_model, label_prompt, simple_mask, args,output_dir="./mask_temp/",final_mask_list=[],mask_save_name=None):
    """
    Processes attention maps, identifies regions of interest, and computes the final mask.

    Parameters:
    - ls_model: Language-specific model for mask prediction.
    - label_prompt: The prompt used for mask prediction.
    - simple_mask: Initial mask used for processing.
    - args: Arguments object containing configurations like height, width, save paths, and prompts.

    Returns:
    - final_mask (np.ndarray): The computed final mask.
    """

    # Convert the mask to binary and resize to match the original image dimensions.
    simple_mask=simple_mask.cpu().detach().numpy()
    binary_mask_zeros = (simple_mask == 0)

    ### dilate the binary_mask_zeros
    mask_size = np.max([binary_mask_zeros.shape[0], binary_mask_zeros.shape[1]])
    ksize = 1 * int(mask_size / 23)
    structure = np.ones((ksize, ksize), dtype=bool)  # You can adjust the size if needed
    binary_mask_zeros = binary_dilation(binary_mask_zeros, structure=structure)



    desired_height = args.height
    desired_width = args.width

    # Resize the binary mask to the original dimensions.
    dilated_mask = cv2.resize(
        binary_mask_zeros.astype(np.uint8),
        (desired_width, desired_height),
        interpolation=cv2.INTER_NEAREST
    )
    dilated_mask = (dilated_mask > 0).astype(np.uint8)



    # Find the bounding box of the region of interest (non-zero region in the mask).
    rows, cols = np.where(dilated_mask)
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("No region with value 0 found in the mask.")
    top, bottom = rows.min(), rows.max()
    left, right = cols.min(), cols.max()

    # Crop the original image using the bounding box.
    image_original = np.array(Image.open(args.image_path).convert("RGB").resize((args.width, args.height), Image.BICUBIC))
    cropped_image_original = Image.fromarray(image_original[top:bottom + 1, left:right + 1])

    image_generated = np.array(Image.open(args.save_path).convert("RGB").resize((args.width, args.height), Image.BICUBIC))
    cropped_image_generated = Image.fromarray(image_generated[top:bottom + 1, left:right + 1])


    # Use the LSAM model to predict the mask for the cropped image.
    original_results = ls_model.predict(
        [cropped_image_original],
        [label_prompt],
        box_threshold=0.3
    )[0]['masks']

    # Use the LSAM model to predict the mask for the cropped image.
    full_results = ls_model.predict(
        [cropped_image_generated],
        [label_prompt],
        box_threshold=0.1
    )[0]['masks']


    # Extract the mask from the results (assuming the first mask is the desired one).

    pass_flag = True
    if len(original_results) == 0:

        cropped_mask = full_results[0]

        # Resize the cropped mask back to the cropped region's size.
        resized_cropped_mask = cv2.resize(
            cropped_mask,
            (right - left + 1, bottom - top + 1),
            interpolation=cv2.INTER_NEAREST
        )

        # Create an empty mask with the same size as the original image.
        final_mask = np.zeros((desired_height, desired_width), dtype=np.uint8)

        # Map the resized cropped mask back to its original position in the full-sized mask.
        final_mask[top:bottom + 1, left:right + 1] = resized_cropped_mask

    elif len(original_results)>0:

        source_mask_list = []
        target_mask_list = []

        ## for source image
        for kk in range(len(original_results)):
            cropped_mask = original_results[kk]
            # Resize the cropped mask back to the cropped region's size.
            resized_cropped_mask = cv2.resize(
                cropped_mask,
                (right - left + 1, bottom - top + 1),
                interpolation=cv2.INTER_NEAREST)
            # Create an empty mask with the same size as the original image.
            original_mask = np.zeros((desired_height, desired_width), dtype=np.uint8)
            # Map the resized cropped mask back to its original position in the full-sized mask.
            original_mask[top:bottom + 1, left:right + 1] = resized_cropped_mask
            source_mask_list.append(original_mask)

        source_mask_list.extend(final_mask_list)


        ## for target image
        for kk in range(len(full_results)):
            cropped_mask = full_results[kk]
            # Resize the cropped mask back to the cropped region's size.
            resized_cropped_mask = cv2.resize(
                cropped_mask,
                (right - left + 1, bottom - top + 1),
                interpolation=cv2.INTER_NEAREST)
            # Create an empty mask with the same size as the original image.
            target_mask = np.zeros((desired_height, desired_width), dtype=np.uint8)
            # Map the resized cropped mask back to its original position in the full-sized mask.
            target_mask[top:bottom + 1, left:right + 1] = resized_cropped_mask
            target_mask_list.append(target_mask)

        print('filtering process')
        print(len(target_mask_list))
        print(len(source_mask_list))
        for kk in range(len(target_mask_list)):
            final_mask = target_mask_list[kk]
            pass_flag=True
            for dd in range(len(source_mask_list)):
                if source_mask_list[dd] is not None:
                    if (((final_mask * source_mask_list[dd]).sum()) / (source_mask_list[dd].sum())) > 0.5:
                        # Skip to the next iteration of the outer loop
                        final_mask=None
                        print('filter')
                        pass_flag=False
                        break  # Exit the inner loop
            if pass_flag:
                break


    if pass_flag==False:
        raise ValueError("Fail to find mask for "+str(label_prompt))

    # Generate a unique filename
    ind = len(os.listdir(output_dir))
    if mask_save_name==None:
        mask_file = f"mask_{ind}.png"
    else:
        mask_file = 'LS_'+str(mask_save_name)+'.png'
    mask_path = os.path.join(output_dir, mask_file)

    # Create the final binary mask and save it
    mask_output = np.where(final_mask != 0, 255, 0).astype(np.uint8)
    cv2.imwrite(mask_path, mask_output)


    return mask_path,final_mask
