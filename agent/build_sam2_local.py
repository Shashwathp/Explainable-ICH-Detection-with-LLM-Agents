import logging
import os
import torch
from omegaconf import OmegaConf
import yaml

def build_sam2(config_file, ckpt_path=None, device="cuda"):
    # Load config directly
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to OmegaConf
    cfg = OmegaConf.create(config)
    
    # Initialize model based on config
    from sam2.modeling.sam2_base import SAM2Base
    model = SAM2Base(
        image_encoder=cfg.model.image_encoder,
        memory_attention=cfg.model.memory_attention,
        memory_encoder=cfg.model.memory_encoder,
        num_maskmem=cfg.model.num_maskmem,
        image_size=cfg.model.image_size,
    )
    
    # Load checkpoint
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        model.load_state_dict(sd)
        logging.info("Loaded checkpoint successfully")
    
    model = model.to(device)
    model.eval()
    
    return model