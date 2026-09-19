"""AnyText 단일 말풍선 원본 출력 확인용."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from PIL import Image
from mangatrans.config import load_config
from mangatrans.render.anytext_renderer import AnyTextRenderer
cfg = load_config()
r = AnyTextRenderer(cfg)
pipe = r._load()
S = 512
crop = Image.new("RGB", (S, S), (255, 255, 255))
mask = Image.new("RGB", (S, S), (255, 255, 255))
from PIL import ImageDraw
ImageDraw.Draw(crop).ellipse([60, 120, 452, 392], outline=(0, 0, 0), width=6)
ImageDraw.Draw(mask).rectangle([130, 200, 380, 250], fill=(0, 0, 0))
ImageDraw.Draw(mask).rectangle([130, 270, 380, 320], fill=(0, 0, 0))
out = pipe('black and white manga page, speech bubble with clean printed Korean text "여기서 알바해?" "할머니 가게라", high contrast',
           negative_prompt="blurry", num_inference_steps=20, mode="edit", draw_pos=mask, ori_image=np.array(crop),
           guidance_scale=9.0, generator=torch.Generator(device="cpu").manual_seed(0)).images[0]
out.save("output/smoke/anytext_probe.png"); print("saved", out.size)
