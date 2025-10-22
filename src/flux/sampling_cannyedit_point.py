import math
from typing import Callable
import numpy as np
import cv2
from PIL import Image
import copy
import time
import torch
from einops import rearrange, repeat
from torch import Tensor
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin
from torchvision import transforms
import random
from typing import Union, List, Dict
from typing import Union
from tqdm import tqdm
from .model import Flux
from .modules.conditioner import HFEmbedder



def get_image_tensor(image,
                     height: int,
                     width: int,
                     device: torch.device,
                     dtype: torch.dtype,
                     ):
    # transforms used for preprocessing dataset
    train_transforms = transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image_tensor = train_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(dtype)
    image_tensor = image_tensor.to(device)
    return image_tensor


def get_image_mask(img, height: int, width: int, device: torch.device, dtype: torch.dtype, ):
    img = np.array(img).astype(np.float32)
    if len(img.shape) == 3:
        img = img[:, :, 0]

    if np.max(img) > 128:
        img = img / 255

    img[img > 0.5] = 1.0
    img[img <= 0.5] = 0.0
    img = img * 255.0

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    img = Image.fromarray(img.astype("uint8")).convert("L")

    resize = transforms.Resize((height, width))
    img = resize(img)
    toT = transforms.ToTensor()
    img = toT(img)
    img[img != 0] = 1
    img = img.unsqueeze(0)
    img = img.to(dtype)
    img = img.to(device)
    return img


def get_noise(
        num_samples: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )





def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: Union[str, List[str]]) -> Dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)


    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)


    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }



def get_lin_function(
        x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_schedule(
        num_steps: int,
        image_seq_len: int,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

# Sigmoid function with adjustable steepness
def sigmoid(x, steepness=10):
    return 1 / (1 + torch.exp(-steepness * (x - 0.5)))



def denoise(
        model: Flux,
        # model input
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        neg_txt: Tensor,
        neg_txt_ids: Tensor,
        neg_vec: Tensor,
        # sampling parameters
        timesteps: list[float],
        guidance: float = 4.0,
        true_gs=1,
        timestep_to_start_cfg=0,
        # ip-adapter parameters
        image_proj: Tensor = None,
        neg_image_proj: Tensor = None,
        ip_scale: Union[Tensor, float] = 1.0,
        neg_ip_scale: Union[Tensor, float]  = 1.0
):
    i = 0
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale,
            )
            pred = neg_pred + true_gs * (pred - neg_pred)
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img


def denoise_fireflow(
        model: Flux,
        # model input
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        # sampling parameters
        timesteps: list[float],
        inverse,
        info,
        guidance: float = 4.0
):


    if inverse:
        timesteps = timesteps[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    next_step_velocity = None
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False

        if inverse==True:

            if next_step_velocity is None:

                block_res_samples = info['controlnet'](
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=info['controlnet_cond'],
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )

                pred = model(
                    img=img,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    block_controlnet_hidden_states=[ij * info['controlnet_gs'] for ij in block_res_samples]

                )


            else:
                pred = next_step_velocity

            img_mid = img + (t_prev - t_curr) / 2 * pred

            t_vec_mid = torch.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype, device=img.device)
            info['second_order'] = True


            block_res_samples = info['controlnet'](
                img=img_mid,
                img_ids=img_ids,
                controlnet_cond=info['controlnet_cond'],
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_mid,
                guidance=guidance_vec,
            )


            pred_mid = model(
                img=img_mid,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_mid,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[ij * info['controlnet_gs'] for ij in block_res_samples]

            )

            next_step_velocity = pred_mid
            img = img + (t_prev - t_curr) * pred_mid
            info[t_curr]=img

    return img, info


def process_mask(input_mask,height,width,latent_image, stage,kernel_size=1,):
    """
    Process the input mask and return processed_mask, dilated_mask, and flattened_mask.

    Args:
        input_mask (torch.Tensor or None): Input mask tensor or None.
        height (int): Height to be used for processing.
        width (int): Width to be used for processing.
        latent_image (torch.Tensor): Source image latent tensor (used for dtype).
        kernel_size (int): Size of the dilation kernel (default is 1).

    Returns:
        tuple: (processed_mask, dilated_mask, flattened_mask)
    """
    # Initialize the processed mask based on the input mask
    if input_mask is None:
        processed_mask = torch.ones((1, int(height / 16) * int(width / 16), 1))
    else:
        processed_mask = copy.deepcopy(input_mask)
    # Ensure processed_mask has the correct dtype and is on GPU
    processed_mask = processed_mask.to(latent_image.dtype).cuda()
    original_mask = copy.deepcopy(processed_mask)
    # Convert processed_mask to numpy and prepare for dilation
    processed_mask_np = (1 - copy.deepcopy(processed_mask).float()).cpu().detach().numpy()
    processed_mask_np = np.squeeze(processed_mask_np).reshape(int(height / 16), int(width / 16))
    # Kernel size and number of iterations for dilation
    iterations = 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform dilation (currently commented out in the original code)
    if stage == 'stage_refine':dilated_mask_np_larger=cv2.dilate(processed_mask_np, (5 * int(height / 512), 5* int(height / 512)), iterations=iterations)
    else: dilated_mask_np_larger = processed_mask_np  # Example: cv2.dilate(processed_mask_np, (4 * int(height / 512), 4 * int(height / 512)), iterations=iterations)
    dilated_mask_np = processed_mask_np  # Example: cv2.dilate(processed_mask_np, kernel, iterations=iterations)
    # Convert dilated masks back to torch tensors
    dilated_mask = torch.tensor(dilated_mask_np, dtype=torch.float32).flatten().unsqueeze(1).cuda()
    dilated_mask_larger = torch.tensor(dilated_mask_np_larger, dtype=torch.float32).flatten().unsqueeze(1).cuda()
    # Update processed_mask and dilated_mask_larger
    processed_mask = 1 - dilated_mask
    dilated_mask = 1 - dilated_mask_larger
    # Compute flattened_mask
    flattened_mask = (1 - processed_mask).flatten()
    return processed_mask, dilated_mask, flattened_mask






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


def denoise_cannyedit_point(
        model: Flux,
        controlnet: None,
        source_image_latent: Tensor,
        source_image_latent_rg:Tensor,
        img: Tensor,
        img_ids: Tensor,
        ## source prompt-related embeddings
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        ## local prompt 1-related embeddings
        txt2: Tensor,
        txt_ids2: Tensor,
        vec2: Tensor,
        ## target prompt-related embeddings
        txt3: Tensor,
        txt_ids3: Tensor,
        vec3: Tensor,
        ## additional local prompts-related embeddings
        txt_addition: list[Tensor],
        txt_ids_addition: list[Tensor],
        vec_addition: list[Tensor],
        ## negative prompt-related embeddings
        neg_txt: Tensor,
        neg_txt_ids: Tensor,
        neg_vec: Tensor,
        local_mask,
        local_mask_addition,
        controlnet_cond,
        # sampling parameters
        timesteps: list[float],
        guidance: float = 4.0,
        true_gs=1,
        controlnet_gs=0.7,
        controlnet_gs2=0.5,
        timestep_to_start_cfg=0,
        # ip-adapter parameters
        image_proj: Tensor = None,
        neg_image_proj: Tensor = None,
        ip_scale: Union[Tensor, float]  = 1,
        neg_ip_scale: Union[Tensor, float]  = 1,
        seed=random.randint(0, 99999),
        generate_save_path=None,
        inversion_save_path=None,
        stage='stage_generate_point',
        norm_softmask=True
):
    regen=0
    if 'regen' in stage:regen=1
    if 'stage_generate' in stage:stage='stage_generate'
    if 'stage_refine' in stage:stage='stage_refine'
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    t_length = len(timesteps)
    if 'stage_generate' in stage:
        info_generate={}
    if generate_save_path != None and stage == 'stage_refine':
        print('load previous generation latents')
        info_generate = np.load(generate_save_path, allow_pickle=True).item()

    # time_to_start = 2
    if  'stage_generate' in stage:
        time_to_start = 2
    if stage == 'stage_refine':
        time_to_start =5
    i = 0

    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1,
                               desc="CannyEdit Denoising Steps"):


        if i==0:
            #-------------------------Inversion-------------------------------------------------------------------------
            if  stage=='stage_generate' and regen==0:
                print('Inversion Start')
                timesteps_inv = timesteps#[time_to_start-2:]
                info = {}
                info['controlnet_cond'] = controlnet_cond
                info['controlnet'] = controlnet
                info['controlnet_gs'] = controlnet_gs2
                start_time = time.time()
                z, info = denoise_fireflow(model, source_image_latent_rg, img_ids, txt, txt_ids, vec, timesteps_inv, guidance=1,
                                           inverse=True, info=info)
                end_time = time.time()
                if inversion_save_path!=None and 'stage_generate' in stage:
                    np.save(inversion_save_path, info)
                print('Inversion End')
            if inversion_save_path!=None and (stage == 'stage_refine' or regen==1):
                print('load previous inversion results')
                info = np.load(inversion_save_path,allow_pickle=True).item()
                info_key_list=list(info.keys())
                def is_strictly_increasing(lst):
                    return all(isinstance(x, int) for x in lst) and all(lst[i] < lst[i + 1] for i in range(len(lst) - 1))
                if is_strictly_increasing(info_key_list):
                    info_back=copy.deepcopy(info)
                    info={}
                    timesteps_transform={}
                    for lk in info_back:
                        info[timesteps[int(lk)]]=info_back[lk]
            # -----------------------End of Inversion----------------------------------------------------------------------

            #---------------------Processing mask---------------------------------------------------------------------
            bs, c, h, w = source_image_latent.shape
            H_use = int(h * 8)
            W_use = int(w * 8)
            source_image_latent = rearrange(source_image_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            source_image_latent = repeat(source_image_latent, "1 ... -> bs ...", bs=bs)
            source_image_latent = source_image_latent.cuda()
            ## Set text embeddings (conds)
            ### len(local_mask_add_proceed)=number of additional local edit prompts + 1 (the first local edit prompt) + 1 (target prompt)
            conds = [None] * (len(local_mask_addition)+2)
            ## the first local prompt and mask for the first local edit region
            conds[1] = txt2
            ## the additional local prompts and their corresponding local edit regions
            for indd in range(len(local_mask_addition)):
                conds[2+indd] = txt_addition[indd]
            ## the target prompt and its mask see the whole image
            conds[0] = txt3

            if stage=='stage_refine':
                local_mask1_proceed, local_mask1_dilate, local_mask1_flat = process_mask(local_mask, H_use, W_use, source_image_latent,stage=stage, kernel_size=1)
                ## process the additional local masks
                local_mask_add_proceed=[]
                local_mask_add_dilate=[]
                local_mask_add_flat=[]
                local_mask_all_dilate=[]
                local_mask_all_dilate.append(local_mask1_dilate)
                if local_mask_addition!=[]:
                    for local_mask1 in local_mask_addition:
                        local_mask2_proceed, local_mask2_dilate, local_mask2_flat = process_mask(local_mask1, H_use, W_use,source_image_latent,stage=stage, kernel_size=1)
                        local_mask_add_proceed.append(local_mask2_proceed)
                        local_mask_add_dilate.append(local_mask2_dilate)
                        local_mask_add_flat.append(local_mask2_flat)
                        local_mask_all_dilate.append(local_mask2_dilate)

                ### initialize the mask (union_mask) used for canny control relaxation and blending, where the value inside the union of the edit regions is 0 and is 1 elsewhere.
                if local_mask_addition==[]:
                    union_mask= local_mask1_dilate
                elif local_mask_addition!=[]:
                    union_inverted = 1 - local_mask1_dilate
                    for mask_dilate in local_mask_add_dilate:
                        mask_dilate_inverted = 1-mask_dilate
                        union_inverted = torch.logical_or(union_inverted.bool(), mask_dilate_inverted.bool())
                    union_inverted=union_inverted.int()
                    union_mask = 1 - union_inverted


                ## Set masks used in attention computation
                ### len(local_mask_add_proceed)=number of additional local edit prompts + 1 (the first local edit prompt) + 1 (target prompt)
                masks = [None] * (len(local_mask_add_proceed)+2)
                masks[1] = 1 - local_mask1_proceed.flatten().unsqueeze(1).repeat(1, conds[1].size(1))
                ## the additional local prompts and their corresponding local edit regions
                for indd in range(len(local_mask_add_proceed)):
                    masks[2+indd] = 1 - local_mask_add_proceed[indd].flatten().unsqueeze(1).repeat(1, conds[2+indd].size(1))
                ## the target prompt and its mask see the whole image
                masks[0] = torch.ones_like(masks[1])


            elif stage=='stage_generate':
                center=local_mask
                local_mask1_soft_mask = create_soft_mask(torch.ones((1, int(H_use / 16) * int(W_use / 16), 1)), center, int(H_use / 16), int(W_use / 16)).flatten().unsqueeze(1).cuda()
                local_softmask_all = []
                local_softmask_all.append(local_mask1_soft_mask)

                if local_mask_addition!=[]:
                    for local_mask1 in local_mask_addition:
                        center = local_mask1
                        local_mask2_soft_mask = create_soft_mask(torch.ones((1, int(H_use / 16) * int(W_use / 16), 1)), center, int(H_use / 16),int(W_use / 16)).flatten().unsqueeze(1).cuda()
                        local_softmask_all.append(local_mask2_soft_mask)

                ###Soft mask used in canny relaxation and blending
                if local_mask_addition==[]:
                    soft_mask_union= local_mask1_soft_mask
                else:
                    soft_mask_union= local_mask1_soft_mask
                    for kkk in local_softmask_all[1:]:
                        soft_mask11 = kkk
                        soft_mask_union=torch.min(soft_mask_union,soft_mask11)
                    soft_mask_union=(soft_mask_union-soft_mask_union.min())/(soft_mask_union.max()-soft_mask_union.min())


            # ------------------End of processing mask------------------------------------------------------------------

            # ------------------Handle attention mask-------------------------------------------------------------------

            regional_embeds = torch.cat(conds, dim=1)
            encoder_seq_len = regional_embeds.shape[1]
            hidden_seq_len = source_image_latent.shape[1]
            txt_ids_region = torch.zeros(regional_embeds.shape[1], 3).to(device=txt_ids.device,dtype=txt_ids.dtype).unsqueeze(0)
            num_of_regions = len(conds)
            each_prompt_seq_len = encoder_seq_len // num_of_regions


            if 'stage_generate' in stage:
                local_mask1_soft_mask=(1-local_mask1_soft_mask.flatten().unsqueeze(1).repeat(1, conds[1].size(1)))*1.
                local_mask1_soft_mask = torch.where(local_mask1_soft_mask < 0.75, float('-inf'), local_mask1_soft_mask)
                # Mask valid values (those not -inf)
                valid_mask = local_mask1_soft_mask != float('-inf')
                valid_values = local_mask1_soft_mask[valid_mask]
                # Normalize valid values between 0 and 1
                min_val = valid_values.min()
                max_val = valid_values.max()
                if min_val.item() < 1:
                    normalized_values = 0.35+(valid_values - min_val) / (max_val - min_val)
                    # Assign normalized values back to the tensor
                    local_mask1_soft_mask[valid_mask] = normalized_values
                for kkk1 in range(len(local_softmask_all)):
                    local_mask2_soft_mask = copy.deepcopy(local_softmask_all[kkk1])
                    thres=0.85
                    local_mask2_soft_mask = (1 - local_mask2_soft_mask.flatten().unsqueeze(1).repeat(1, conds[1].size(1))) * 1.  # *10000
                    local_mask2_soft_mask = torch.where(local_mask2_soft_mask < thres, float('-inf'),local_mask2_soft_mask)

                    #-------------------------------------------------
                    if norm_softmask:
                        ## Mask valid values (those not -inf)
                        valid_mask = local_mask2_soft_mask != float('-inf')
                        valid_values = local_mask2_soft_mask[valid_mask]
                        ## Normalize valid values between 0 and 1
                        min_val = valid_values.min()
                        max_val = valid_values.max()
                        if min_val.item()<1:
                            normalized_values = thres+ (valid_values - min_val) / (max_val - min_val)
                            # # Assign normalized values back to the tensor
                            local_mask2_soft_mask[valid_mask] = normalized_values
                    # -------------------------------------------------
                    local_softmask_all[kkk1]=local_mask2_soft_mask
                    del local_mask2_soft_mask


            # ================================
            ## T2T, T2I and I2T attention mask
            ## Each text can only see itself
            ## Local prompt can only see/be seen by the local edit region
            ## Target prompt can see/be seen by the whole image

            # initialize attention mask
            if stage=='stage_generate' or num_of_regions==2:dtype1=torch.float
            else:dtype1=torch.bool
            regional_attention_mask = torch.zeros((encoder_seq_len + hidden_seq_len, encoder_seq_len + hidden_seq_len), device=conds[0].device,dtype=dtype1)


            for ij in range(num_of_regions):
                # t2t mask txt attends to itself
                regional_attention_mask[ij * each_prompt_seq_len:(ij + 1) * each_prompt_seq_len, ij * each_prompt_seq_len:(ij + 1) * each_prompt_seq_len] = True
                if 'stage_generate' in stage:
                    ## the whole image can see/be seen by the global target prompt
                    if ij==0:
                        regional_attention_mask[ij * each_prompt_seq_len:(ij + 1) * each_prompt_seq_len, encoder_seq_len:] = True
                        regional_attention_mask[encoder_seq_len:, ij * each_prompt_seq_len:(ij + 1) * each_prompt_seq_len] = True
                    else:
                        if num_of_regions==2:
                            regional_attention_mask[ij * each_prompt_seq_len:(ij + 1) * each_prompt_seq_len,encoder_seq_len:] = local_mask1_soft_mask.transpose(-1, -2)
                            regional_attention_mask[encoder_seq_len:,ij * each_prompt_seq_len:(ij + 1) * each_prompt_seq_len] = local_mask1_soft_mask
                        else:
                            regional_attention_mask[ij * each_prompt_seq_len:(ij + 1) * each_prompt_seq_len, encoder_seq_len:] =local_softmask_all[ij-1].transpose(-1, -2)
                            regional_attention_mask[encoder_seq_len:,ij * each_prompt_seq_len:(ij + 1) * each_prompt_seq_len] =local_softmask_all[ij-1]
                elif stage == 'stage_refine':
                    regional_attention_mask[ij * each_prompt_seq_len:(ij + 1) * each_prompt_seq_len,encoder_seq_len:] = masks[ij].transpose(-1, -2)
                    regional_attention_mask[encoder_seq_len:,ij * each_prompt_seq_len:(ij + 1) * each_prompt_seq_len] =  masks[ij]

            regional_attention_mask[encoder_seq_len:, encoder_seq_len:]=torch.ones_like(regional_attention_mask[encoder_seq_len:, encoder_seq_len:])


        # ------------------End of Handle attention mask-------------------------------------------------------------------

        apply_local_point = 0.7
        apply_extenda_point = 0.8

        if timesteps[i] not in info:tempp=timesteps[i+1]
        else:tempp=timesteps[i]

        if i >= time_to_start:
            if i == time_to_start:
                img = info[timesteps[i+4]]
                print('seed')
                print(seed)

                # ------------------Reinitialize each local edit region--------------------------------------------------

                if  'stage_generate' in stage:
                    #---------------------------
                    noise1 = img
                    mean = torch.mean(noise1)
                    std = torch.std(noise1)
                    noise1 = (noise1 - mean) / std
                    #---------------------------
                    ##Initialization
                    x2 = get_noise(1, H_use, W_use, device='cuda', dtype=torch.bfloat16, seed=seed)
                    bs, c, h, w = x2.shape
                    x2 = rearrange(x2, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                    x2 = repeat(x2, "1 ... -> bs ...", bs=bs)
                    seed += 1
                    noise2 = x2
                    mean = torch.mean(noise2)
                    std = torch.std(noise2)
                    noise2 = (noise2 - mean) / std

                    center=local_mask
                    soft_mask_local = create_soft_mask( torch.ones((1, int(H_use / 16) * int(W_use / 16), 1)).flatten(), center, int(H_use / 16),
                                                 int(W_use / 16)).flatten().cuda().unsqueeze(0).unsqueeze(-1)
                    soft_mask_local=1-soft_mask_local

                    if num_of_regions>2:
                        for local_mask1 in local_mask_addition:
                            center = local_mask1
                            soft_mask_local11 = create_soft_mask(torch.ones((1, int(H_use / 16) * int(W_use / 16), 1)).flatten(), center, int(H_use / 16),
                                                               int(W_use / 16)).flatten().cuda().unsqueeze(0).unsqueeze(-1)
                            soft_mask_local11 = 1 - soft_mask_local11
                            soft_mask_local = torch.max(soft_mask_local11, soft_mask_local)
                        soft_mask_local=soft_mask_local**1.2

                    # Add the random noise to the inverted noise scaled by the soft mask
                    combined_noise = (1-soft_mask_local)*noise1 +  noise2*soft_mask_local
                    mean = torch.mean(combined_noise)
                    std = torch.std(combined_noise)
                    combined_noise = (combined_noise - mean) / std
                    img= combined_noise.to(torch.bfloat16)




                elif stage == 'stage_refine':

                    x2 = get_noise(1, H_use, W_use, device='cuda', dtype=torch.bfloat16, seed=seed)
                    bs, c, h, w = x2.shape
                    x2 = rearrange(x2, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                    x2 = repeat(x2, "1 ... -> bs ...", bs=bs)
                    seed+=1
                    if num_of_regions==2:prev_generate,aaaa,bbbb = info_generate[i + 2],0.15,0.15
                    else:prev_generate,aaaa,bbbb = info_generate[i + 4],0.15,0.35


                    img[:, local_mask1_flat.bool(), :] =  aaaa*img[:, local_mask1_flat.bool(), :]+(1-aaaa-bbbb)*x2[:, local_mask1_flat.bool(), :]+bbbb*prev_generate[:, local_mask1_flat.bool(), :] #aaaa*prev_generate[:, local_mask1_flat.bool(), :]+(1-aaaa)*x2[:, local_mask1_flat.bool(), :]
                    for local_mask2_flat in local_mask_add_flat:

                        x3 = get_noise(1, H_use, W_use, device='cuda', dtype=torch.bfloat16, seed=seed)
                        bs, c, h, w = x3.shape
                        x3 = rearrange(x3, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                        x3 = repeat(x3, "1 ... -> bs ...", bs=bs)
                        seed += 1
                        img[:, local_mask2_flat.bool(), :] = aaaa*img[:, local_mask2_flat.bool(), :]+(1-aaaa-bbbb)*x2[:, local_mask2_flat.bool(), :]+bbbb*prev_generate[:, local_mask2_flat.bool(), :]  #aaaa*prev_generate[:, local_mask2_flat.bool(), :]+(1-aaaa)*x3[:, local_mask2_flat.bool(), :]


                # ------------------END of Reinitialize each local edit region-------------------------------------------


            #================================== Start Denoising =================================================

            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            imgg=info[tempp]

            block_res_samples = controlnet(
                img=imgg,
                img_ids=img_ids,
                controlnet_cond=controlnet_cond,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )
            # ------------------Selective Canny Masking-------------------------------------------------------------------
            soft_masks = []
            # Generate a Gaussian soft mask for each tensor
            if 'stage_generate' in stage:
                for tensor in block_res_samples:
                    if num_of_regions==2:
                        soft_masks.append((soft_mask_union ** 1.05).to(device=tensor.device, dtype=tensor.dtype))
                    else:soft_masks.append((soft_mask_union**1.2).to(device=tensor.device, dtype=tensor.dtype))
                block_res_samples = [block_res_samples[i] * soft_masks[i] for i in range(len(block_res_samples))]
            elif stage == 'stage_refine':
                soft_masks = []
                for tensor in block_res_samples:
                    soft_masks.append(union_mask.to(device=tensor.device, dtype=tensor.dtype))  # Add to list
                block_res_samples = [block_res_samples[i] * soft_masks[i] for i in range(len(block_res_samples))]
            # ------------------End of Selective Canny Masking--------------------------------------------------------------



            ### Stage 1: Regional Denoising
            if i < int(apply_local_point*t_length):
                attention_kwargs={}
                #*********************************************
                # union of all local mask
                if 'stage_generate' in stage:
                    attention_kwargs['union_mask'] = None
                    attention_kwargs['local_mask_all_dilate']=[]
                elif stage == 'stage_refine':
                    attention_kwargs['union_mask'] = union_mask
                    ## input each local mask
                    attention_kwargs['local_mask_all_dilate']=local_mask_all_dilate
                ## number of local edit regions
                attention_kwargs['num_edit_region'] = 1 + len(txt_addition)
                ## attention_mask
                if 'stage_generate' in stage:
                    attention_kwargs['regional_attention_mask'] = regional_attention_mask.float()#.bool()
                elif stage == 'stage_refine':
                    attention_kwargs['regional_attention_mask'] = regional_attention_mask.bool()
                if num_of_regions == 2:
                    ## the attention between the local text promt and local edit region
                    attention_kwargs['local_t2i_strength'] = 1.2
                    ## the attention within each edit region
                    attention_kwargs['local_i2i_strength'] = 1.0
                else:
                    ## the attention between the local text promt and local edit region
                    attention_kwargs['local_t2i_strength'] = 1.3
                    ## the attention within each edit region
                    attention_kwargs['local_i2i_strength'] = 1.05
                ## attention between the target prompt and the whole image
                attention_kwargs['context_t2i_strength'] = 1
                ## attention between each edit region and other regions
                attention_kwargs['local2out_i2i_strength'] = 1
                attention_kwargs['image_size'] = int(H_use / 16) * int(W_use / 16)
                if 'stage_generate' in stage and i==30:
                    attention_kwargs['save_attn']=True
                #*********************************************


                if i <= int(t_length * apply_extenda_point):controlnet_control = [ij * controlnet_gs2 for ij in block_res_samples]
                else:controlnet_control = None

                pred = model(
                    img=img,
                    img_ids=img_ids,
                    txt=regional_embeds,
                    txt_ids=txt_ids_region,
                    y=torch.zeros_like(vec3),
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    block_controlnet_hidden_states=controlnet_control,
                    image_proj=image_proj,
                    ip_scale=ip_scale,
                    attention_kwargs=attention_kwargs
                )

            ### Stage 2: Normal Denoising
            else:
                attention_kwargs={}
                if i <= int(t_length * apply_extenda_point):controlnet_control=[ij * controlnet_gs2 for ij in block_res_samples]
                else:controlnet_control = None

                pred = model(
                    img=img,
                    img_ids=img_ids,
                    txt=txt3,
                    txt_ids=txt_ids3,
                    y=vec3,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    block_controlnet_hidden_states=controlnet_control,
                    image_proj=image_proj,
                    ip_scale=ip_scale,
                    attention_kwargs=attention_kwargs
                )



            img = img + (t_prev - t_curr) * pred

            ## Cyclical Blending
            if  'stage_generate' in stage:
                soft_mask22 = copy.deepcopy(soft_mask_union)**1.5
                img = (soft_mask22).to(torch.bfloat16) * info[tempp] + (1 - soft_mask22).to(torch.bfloat16) * img

            elif stage == 'stage_refine':
                if num_of_regions == 2:
                    if i < 5 or (i <= 20 and i % 5 == 0):
                        img = torch.where(union_mask == 1, 0.5 * info[tempp] + 0.5 * img, img)
                    elif (i <= 40 and i % 10 == 0):
                        img = torch.where(union_mask == 1, 0.3 * info[tempp] + 0.7 * img, img)
                else:
                    if i < 5 or (i <= 20 and i % 5 == 0):
                        img = torch.where(union_mask == 1, 0.5 * info[tempp] + 0.5 * img,img)
                    elif (i <= 40 and i % 10 == 0):
                        img = torch.where(union_mask == 1, 0.2 * info[tempp] + 0.8 * img,img)

            if generate_save_path!=None:
                info_generate[i] = img

        i += 1

    if generate_save_path!=None:
        np.save(generate_save_path, info_generate)

    return img



def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
