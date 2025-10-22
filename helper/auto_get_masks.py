
## code for refining the mask from the user-given binary mask
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import cv2
import matplotlib


def auto_get_mask(ls_model, label_prompt, simple_mask, args,output_dir="./mask_temp/"):
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
    binary_mask_zeros = (simple_mask.numpy() == 0)

    ### dilate the binary_mask_zeros
    mask_size = np.max([binary_mask_zeros.shape[0], binary_mask_zeros.shape[1]])
    ksize = 2 * int(mask_size / 23)
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

    # plt.imshow(cropped_image_generated)
    # plt.axis('off')  # Hide axes
    # plt.title('Cropped Image')
    # plt.show()

    # Extract the mask from the results (assuming the first mask is the desired one).


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
            for dd in range(len(source_mask_list)):
                # if (((final_mask * source_mask_list[dd]).sum()) / ((right - left + 1) * (bottom - top + 1))) > 0.5:
                if (((final_mask * source_mask_list[dd]).sum()) / (source_mask_list[dd].sum())) > 0.5:
                    # Skip to the next iteration of the outer loop
                    final_mask=None
                    print('filter')
                    break  # Exit the inner loop

    # Visualize the final mask on top of the image

    plt.figure(figsize=(10, 10))
    plt.imshow(image_generated)
    plt.imshow(final_mask, cmap='jet', alpha=0.5)  # Overlay the mask with transparency
    plt.axis('off')
    plt.title("Final Mask Overlay")
    plt.show()

    # Generate a unique filename
    ind = len(os.listdir(output_dir))
    mask_file = f"mask_{ind}.png"
    mask_path = os.path.join(output_dir, mask_file)

    # Create the final binary mask and save it
    mask_output = np.where(final_mask != 0, 255, 0).astype(np.uint8)
    cv2.imwrite(mask_path, mask_output)


    return mask_path
