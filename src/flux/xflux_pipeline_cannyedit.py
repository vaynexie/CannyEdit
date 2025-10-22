from PIL import Image, ExifTags
import numpy as np
import torch
from torch import Tensor
import copy
from einops import rearrange
import uuid
import os
import cv2
import matplotlib.pyplot as plt
from typing import List


from src.flux.modules.layers_cannyedit import (
    SingleStreamBlockProcessor,
    DoubleStreamBlockProcessor,
)


from src.flux.sampling_cannyedit import (
    denoise,
    denoise_cannyedit,
    get_noise,
    get_schedule,
    prepare,
    unpack,
    get_image_tensor,
    get_image_mask)

from src.flux.sampling_cannyedit_removal import (
    denoise_cannyedit_removal)

from src.flux.sampling_cannyedit_point import (
    denoise_cannyedit_point)


from src.flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    load_controlnet,
    load_flow_model_quintized,
    Annotator,
    load_checkpoint,
)
from src.flux.add_tokens import add_tokens

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


def prepare_conditional_inputs(base_input, suffix):
    """
    Helper function to process and restructure the input dictionary.
    """
    result = {}
    for key in ['txt', 'txt_ids', 'vec']:
        result[f"{key}{suffix}"] = base_input[key]
    base_input.pop('img')  # Remove the key from the original dictionary
    base_input.pop('img_ids')  # Remove the key from the original dictionary
    return result


class XFluxPipeline:
    def __init__(self, model_type, device, offload: bool = False):
        self.device = torch.device(device)
        self.offload = offload
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        if "fp8" in model_type:
            self.model = load_flow_model_quintized(model_type, device="cpu" if offload else self.device)
        else:
            self.model = load_flow_model(model_type, device="cpu" if offload else self.device)

        self.image_encoder_path = "openai/clip-vit-large-patch14"
        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False
        self.ip_loaded = False
        self.paint_loaded = False



    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None):
        self.model.to(self.device)
        self.controlnet = load_controlnet(self.model_type, self.device).to(torch.bfloat16)
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)
        self.annotator = Annotator(control_type, self.device)
        self.controlnet_loaded = True
        self.control_type = control_type


    def __call__(self,
                 ## change:add parameters
                 prompt_source: str,
                 prompt_local1: str,
                 prompt_target: str,
                 prompt_local_addition: List[str],
                 image_prompt: Image = None,
                 controlnet_image: Image = None,
                 local_mask=None,
                 local_mask_addition=[],
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 seed: int = 123456789,
                 true_gs: float = 3,
                 control_weight: float = 0.9,
                 control_weight2: float = 0.5,
                 ip_scale: float = 1.0,
                 neg_ip_scale: float = 1.0,
                 neg_prompt: str = '',
                 neg_prompt2: str = '',
                 neg_image_prompt: Image = None,
                 timestep_to_start_cfg: int = 0,
                 image_paint=None,
                 promptA=None,
                 promptB=None,
                 negative_promptA=None,
                 negative_promptB=None,
                 tradeoff=None,
                 mask=None,
                 generate_save_path=None,
                 inversion_save_path=None,
                 stage=None,
                 norm_softmask=True
                 ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)
        image_proj = None
        neg_image_proj = None

        ## change: process the source image
        if self.controlnet_loaded:
            source_image = copy.deepcopy(controlnet_image)
            controlnet_cond = self.annotator(controlnet_image, width, height)
            controlnet_cond = torch.from_numpy((np.array(controlnet_cond) / 127.5) - 1)
            controlnet_cond = controlnet_cond.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(self.device)

        ## change:add parameters
        return self.forward(
            prompt_source,
            prompt_local1,
            prompt_target,
            prompt_local_addition,
            local_mask,
            local_mask_addition,
            width,
            height,
            guidance,
            num_steps,
            seed,
            source_image,
            controlnet_cond,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            control_weight=control_weight,
            control_weight2=control_weight2,
            neg_prompt=neg_prompt,
            neg_prompt2=neg_prompt2,
            image_proj=image_proj,
            neg_image_proj=neg_image_proj,
            ip_scale=ip_scale,
            neg_ip_scale=neg_ip_scale,
            image_paint=image_paint,
            promptA=promptA,
            promptB=promptB,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            tradeoff=tradeoff,
            mask=mask,
            generate_save_path=generate_save_path,
            inversion_save_path=inversion_save_path,
            stage=stage,
            norm_softmask=True

        )

    ## change:add parameters
    def forward(
        self,
        prompt_source,
        prompt_local1,
        prompt_target,
        prompt_local_addition,
        local_mask,
        local_mask_addition,
        width,
        height,
        guidance,
        num_steps,
        seed,
        source_image=None,
        controlnet_cond = None,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
        control_weight = 0.9,
        control_weight2= 0.5,
        neg_prompt="",
        neg_prompt2="",
        image_proj=None,
        neg_image_proj=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
        image_paint=None,
        promptA=None,
        promptB=None,
        negative_promptA=None,
        negative_promptB=None,
        tradeoff=None,
        mask=None,
        generate_save_path=None,
        inversion_save_path=None,
        stage=None,
        norm_softmask=True
    ):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )

        ## change:process source image
        source_image_latent = self.ae.encode(get_image_tensor(source_image, height, width, device=next(self.ae.parameters()).device,dtype=torch.float32)).to(torch.bfloat16)

        # ==============================
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )

        torch.manual_seed(seed)
        ## change:process the more text prompts
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

            inp_cond_im = prepare(t5=self.t5, clip=self.clip, img=source_image_latent, prompt=prompt_source)
            source_image_latent_rg = inp_cond_im['img']


            # Prepare inputs with different prompts
            inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt='a real-world image of ' + prompt_source)
            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)
            neg_inp_cond2 = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt2)
            # Process inp_cond2, local prompt 1
            inp_cond2 = prepare(t5=self.t5, clip=self.clip, img=x, prompt='a real-world image of ' + prompt_local1)
            inp_cond2 = prepare_conditional_inputs(inp_cond2, '2')
            # Process inp_cond3, target prompt
            inp_cond3 = prepare(t5=self.t5, clip=self.clip, img=x, prompt='a real-world image of ' + prompt_target)
            inp_cond3 = prepare_conditional_inputs(inp_cond3, '3')
            # Process additional local prompts
            inp_cond_addition={}
            inp_cond_addition['txt_addition'] = []
            inp_cond_addition['txt_ids_addition'] = []
            inp_cond_addition['vec_addition'] = []
            for pp in prompt_local_addition:
                inp_cond4 = prepare(t5=self.t5, clip=self.clip, img=x, prompt='a real-world image of ' + str(pp))
                inp_cond_addition['txt_addition'].append(inp_cond4['txt'])
                inp_cond_addition['txt_ids_addition'].append(inp_cond4['txt_ids'])
                inp_cond_addition['vec_addition'].append(inp_cond4['vec'])

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            ## change: add parameters
            if self.controlnet_loaded:
                if stage=='stage_generate' or stage=='stage_refine' or stage=='stage_generate_regen':
                    x = denoise_cannyedit(
                        self.model,
                        **inp_cond,
                        **inp_cond2,
                        **inp_cond3,
                        **inp_cond_addition,
                        local_mask=local_mask,
                        local_mask_addition=local_mask_addition,
                        source_image_latent=source_image_latent,
                        source_image_latent_rg=source_image_latent_rg,
                        controlnet=self.controlnet,
                        timesteps=timesteps,
                        guidance=guidance,
                        controlnet_cond=controlnet_cond,
                        timestep_to_start_cfg=timestep_to_start_cfg,
                        neg_txt=neg_inp_cond['txt']*0.5+neg_inp_cond2['txt']*0.5,
                        neg_txt_ids=neg_inp_cond['txt_ids'],
                        neg_vec=neg_inp_cond['vec']*0.5+neg_inp_cond2['vec']*0.5,
                        true_gs=true_gs,
                        controlnet_gs=control_weight,
                        controlnet_gs2=control_weight2,
                        seed=seed,
                        generate_save_path=generate_save_path,
                        inversion_save_path=inversion_save_path,
                        stage=stage
                    )
                if stage=='stage_generate_point' or stage=='stage_refine_point' or stage=='stage_generate_point_regen':
                    x = denoise_cannyedit_point(
                        self.model,
                        **inp_cond,
                        **inp_cond2,
                        **inp_cond3,
                        **inp_cond_addition,
                        local_mask=local_mask,
                        local_mask_addition=local_mask_addition,
                        source_image_latent=source_image_latent,
                        source_image_latent_rg=source_image_latent_rg,
                        controlnet=self.controlnet,
                        timesteps=timesteps,
                        guidance=guidance,
                        controlnet_cond=controlnet_cond,
                        timestep_to_start_cfg=timestep_to_start_cfg,
                        neg_txt=neg_inp_cond['txt']*0.5+neg_inp_cond2['txt']*0.5,
                        neg_txt_ids=neg_inp_cond['txt_ids'],
                        neg_vec=neg_inp_cond['vec']*0.5+neg_inp_cond2['vec']*0.5,
                        true_gs=true_gs,
                        controlnet_gs=control_weight,
                        controlnet_gs2=control_weight2,
                        seed=seed,
                        generate_save_path=generate_save_path,
                        inversion_save_path=inversion_save_path,
                        stage=stage,
                        norm_softmask=True
                    )
                elif stage=='stage_removal' or stage=='stage_removal_regen':
                    x = denoise_cannyedit_removal(
                        self.model,
                        **inp_cond,
                        **inp_cond2,
                        **inp_cond3,
                        **inp_cond_addition,
                        local_mask=local_mask,
                        local_mask_addition=local_mask_addition,
                        source_image_latent=source_image_latent,
                        source_image_latent_rg=source_image_latent_rg,
                        controlnet=self.controlnet,
                        timesteps=timesteps,
                        guidance=guidance,
                        controlnet_cond=controlnet_cond,
                        timestep_to_start_cfg=timestep_to_start_cfg,
                        neg_txt=neg_inp_cond['txt'] * 0.5 + neg_inp_cond2['txt'] * 0.5,
                        neg_txt_ids=neg_inp_cond['txt_ids'],
                        neg_vec=neg_inp_cond['vec'] * 0.5 + neg_inp_cond2['vec'] * 0.5,
                        true_gs=true_gs,
                        controlnet_gs=control_weight,
                        controlnet_gs2=control_weight2,
                        seed=seed,
                        generate_save_path=generate_save_path,
                        inversion_save_path=inversion_save_path,
                        stage=stage
                    )



            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)

            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()



class XFluxSampler(XFluxPipeline):
    def __init__(self, clip, t5, ae, model, device):
        self.clip = clip
        self.t5 = t5
        self.ae = ae
        self.model = model
        self.model.eval()
        self.device = device
        self.controlnet_loaded = False
        self.ip_loaded = False
        self.offload = False
