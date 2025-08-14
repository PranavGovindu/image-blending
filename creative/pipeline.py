from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from rembg import remove

import json
import requests


@dataclass
class GenerationOutput:
    story: str
    character_description: str
    background_description: str
    character_image_path: str
    background_image_path: str
    combined_image_path: str


def generate_texts(seed_prompt: Optional[str]) -> tuple[str, str, str]:
    seed = seed_prompt.strip() if seed_prompt else "A brave explorer enters a hidden valley at dawn."
    
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen2:1.5b-instruct")
    
    prompt = f"""You are a creative writing assistant. Generate a JSON response with:
- story: 2-3 sentence short story from this seed
- character_description: detailed character with visual attributes
- background_description: detailed scene/environment with lighting

Seed: {seed}

Return only JSON:"""

    response = requests.post(
        f"{ollama_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60
    )
    
    text = response.json()["response"]
    import re
    match = re.search(r'\{[\s\S]*\}', text)
    data = json.loads(match.group(0))
    
    return data["story"], data["character_description"], data["background_description"]


def craft_character_prompt(character_description: str) -> str:
    return f"""Professional character portrait: {character_description}. 
Cinematic photography, ultra-detailed, photorealistic rendering. 
Clean subject isolation, transparent background, sharp focus on character. 
DSLR quality, 85mm portrait lens, shallow depth of field."""


def craft_background_prompt(background_description: str) -> str:
    return f"""Cinematic environment shot: {background_description}. 
Wide-angle landscape photography, perfect for character compositing. 
EMPTY SCENE with no people, animals, or figures present. 
Professional lighting setup, detailed atmospheric effects."""






def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def _resize_with_aspect(image: np.ndarray, target_height: int) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = target_height / max(1, height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return resized, scale


def _refine_mask(binary_mask: np.ndarray, feather_px: int = 3) -> np.ndarray:
    mask = binary_mask.copy()
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    if feather_px > 0:
        mask = cv2.GaussianBlur(mask, (feather_px * 2 + 1, feather_px * 2 + 1), 0)
    return mask


def _color_transfer_lab(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    src = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_means, src_stds = cv2.meanStdDev(src)
    tgt_means, tgt_stds = cv2.meanStdDev(tgt)
    tgt_stds = np.where(tgt_stds < 1e-6, 1.0, tgt_stds)
    out = (src - src_means.reshape(1, 1, 3)) * (tgt_stds.reshape(1, 1, 3) / src_stds.reshape(1, 1, 3)) + tgt_means.reshape(1, 1, 3)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _estimate_light_direction(bgr: np.ndarray) -> Tuple[float, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.arctan2(gy, gx)
    mag = np.sqrt(gx * gx + gy * gy)
    s = np.sin(angle)
    c = np.cos(angle)
    mean_s = float(np.sum(s * mag) / (np.sum(mag) + 1e-6))
    mean_c = float(np.sum(c * mag) / (np.sum(mag) + 1e-6))
    dx, dy = mean_c, mean_s
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-3:
        return (-0.6, -0.8)
    return (dx / norm, dy / norm)


def _apply_atmospheric_perspective(bgr: np.ndarray, intensity: float = 0.06) -> np.ndarray:
    if intensity <= 0:
        return bgr
    haze_color = np.array([220, 220, 220], dtype=np.float32)
    out = bgr.astype(np.float32)
    out = out * (1.0 - intensity) + haze_color * intensity
    return _ensure_uint8(out)


def _add_film_grain(bgr: np.ndarray, sigma: float = 4.0, strength: float = 0.08) -> np.ndarray:
    if strength <= 0:
        return bgr
    noise = np.random.normal(0, sigma, bgr.shape).astype(np.float32)
    out = bgr.astype(np.float32) + noise * strength
    return _ensure_uint8(out)


def _add_chromatic_aberration(bgr: np.ndarray, shift_px: int = 1) -> np.ndarray:
    if shift_px <= 0:
        return bgr
    b, g, r = cv2.split(bgr)
    rows, cols = b.shape
    M_right = np.float32([[1, 0, shift_px], [0, 1, 0]])
    M_left = np.float32([[1, 0, -shift_px], [0, 1, 0]])
    r_shifted = cv2.warpAffine(r, M_right, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    b_shifted = cv2.warpAffine(b, M_left, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = cv2.merge([b_shifted, g, r_shifted])
    return out


def _add_rim_light(char_bgr: np.ndarray, mask: np.ndarray, bg_color_sample: Tuple[int, int, int], strength: float = 0.25) -> np.ndarray:
    if strength <= 0:
        return char_bgr
    mask_blur = cv2.GaussianBlur(mask, (7, 7), 0)
    edge = cv2.Laplacian(mask_blur, cv2.CV_32F, ksize=3)
    edge = np.clip(edge, 0, None)
    edge = (edge / (edge.max() + 1e-6) * 255.0).astype(np.uint8)
    edge = cv2.GaussianBlur(edge, (5, 5), 0)
    color = np.array(bg_color_sample, dtype=np.float32)
    edge_3c = cv2.merge([edge, edge, edge]).astype(np.float32) / 255.0
    base = char_bgr.astype(np.float32)
    out = base * (1 - edge_3c * strength) + color * edge_3c * strength
    return _ensure_uint8(out)


def _compose_over(background_bgr: np.ndarray, foreground_bgr: np.ndarray, mask: np.ndarray, center_xy: Tuple[int, int]) -> np.ndarray:
    x_center, y_center = center_xy
    h, w = foreground_bgr.shape[:2]
    y1 = max(0, y_center - h // 2)
    x1 = max(0, x_center - w // 2)
    y2 = min(background_bgr.shape[0], y1 + h)
    x2 = min(background_bgr.shape[1], x1 + w)
    fg_y1 = 0
    fg_x1 = 0
    if y2 - y1 < h:
        fg_y1 = h - (y2 - y1)
    if x2 - x1 < w:
        fg_x1 = w - (x2 - x1)
    roi_bg = background_bgr[y1:y2, x1:x2]
    roi_fg = foreground_bgr[fg_y1:fg_y1 + (y2 - y1), fg_x1:fg_x1 + (x2 - x1)]
    roi_mask = mask[fg_y1:fg_y1 + (y2 - y1), fg_x1:fg_x1 + (x2 - x1)]
    m = (roi_mask.astype(np.float32) / 255.0)[..., None]
    composed = roi_bg.astype(np.float32) * (1 - m) + roi_fg.astype(np.float32) * m
    background_bgr[y1:y2, x1:x2] = _ensure_uint8(composed)
    return background_bgr


def _generate_contact_shadow(mask: np.ndarray, offset: Tuple[float, float], blur: int = 25, opacity: float = 0.6) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    shift_x, shift_y = offset
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    k = blur if blur % 2 == 1 else blur + 1
    blurred = cv2.GaussianBlur(shifted, (k, k), 0)
    shadow = (blurred.astype(np.float32) / 255.0) * (opacity * 255.0)
    return shadow.astype(np.uint8)


class ImageCompositor:
    def composite_images(self, character_rgba_path: str, background_path: str, media_dir: Path) -> str:
        bg = cv2.imread(background_path, cv2.IMREAD_COLOR)
        if bg is None:
            raise RuntimeError("Failed to load background image")

        char_rgba = Image.open(character_rgba_path).convert("RGBA")
        char_np = np.array(char_rgba)
        alpha = char_np[:, :, 3]
        char_rgb = char_np[:, :, :3]
        char_bgr = cv2.cvtColor(char_rgb, cv2.COLOR_RGB2BGR)

        target_h = int(bg.shape[0] * 0.42)
        char_bgr, _ = _resize_with_aspect(char_bgr, target_h)
        mask = cv2.resize(alpha, (char_bgr.shape[1], char_bgr.shape[0]), interpolation=cv2.INTER_AREA)
        mask = _refine_mask(mask, feather_px=3)

        char_bgr = _color_transfer_lab(char_bgr, bg)

        lx, ly = _estimate_light_direction(bg)
        center_sample = bg[bg.shape[0] // 2, bg.shape[1] // 2]
        char_bgr = _add_rim_light(char_bgr, mask, tuple(int(x) for x in center_sample.tolist()), strength=0.18)

        center_xy = (int(bg.shape[1] * 0.26), int(bg.shape[0] * 0.74))

        offset_px = (-int(lx * char_bgr.shape[1] * 0.12), -int(ly * char_bgr.shape[0] * 0.05))
        shadow = _generate_contact_shadow(mask, offset_px, blur=max(21, int(min(char_bgr.shape[:2]) * 0.08)), opacity=0.55)
        shadow_bgr = np.zeros_like(bg)
        shadow_3c = cv2.merge([shadow, shadow, shadow])
        shadow_canvas = _compose_over(shadow_bgr, shadow_3c, shadow, center_xy)
        shadow_alpha = (shadow_canvas[:, :, 0].astype(np.float32) / 255.0)[..., None]
        bg = _ensure_uint8(bg.astype(np.float32) * (1.0 - shadow_alpha * 0.5))

        bg = _compose_over(bg, char_bgr, mask, center_xy)

        bg = _apply_atmospheric_perspective(bg, intensity=0.05)
        bg = _add_film_grain(bg, sigma=3.0, strength=0.06)
        bg = _add_chromatic_aberration(bg, shift_px=1)

        out_path = str((Path(media_dir) / "combined.png").resolve())
        cv2.imwrite(out_path, bg)
        return out_path



def generate_character_image(character_description: str, media_dir: Path) -> str:
    character_raw_path = media_dir / "character_raw.png"
    character_rgba_path = media_dir / "character_rgba.png"

    hf_token = os.environ.get("HF_TOKEN")
    prompt = craft_character_prompt(character_description)
    
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt, "wait_for_model": True}
    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    
    print(f"HF Response status: {response.status_code}")
    print(f"HF Response content: {response.content[:200]}")
    
    from io import BytesIO
    image = Image.open(BytesIO(response.content))
    image = image.convert("RGB").resize((1024, 1024))
    image.save(character_raw_path)
    cutout = remove(Image.open(character_raw_path))
    cutout.save(character_rgba_path)
    return str(character_rgba_path.resolve())


def generate_background_image(background_description: str, media_dir: Path) -> str:
    background_path = media_dir / "background.png"

    hf_token = os.environ.get("HF_TOKEN")
    prompt = craft_background_prompt(background_description)
    
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt, "wait_for_model": True}
    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    
    from io import BytesIO
    image = Image.open(BytesIO(response.content))
    image = image.convert("RGB").resize((1536, 1024))
    image.save(background_path)
    return str(background_path.resolve())


def combine_images(character_path: str, background_path: str, media_dir: Path) -> str:
    compositor = ImageCompositor()
    return compositor.composite_images(character_path, background_path, media_dir)


def orchestrate(seed_prompt: Optional[str], media_dir: Path) -> GenerationOutput:
    
    story, character_desc, background_desc = generate_texts(seed_prompt)

    char_img = generate_character_image(character_desc, media_dir)
    bg_img = generate_background_image(background_desc, media_dir)

    combined_img = combine_images(char_img, bg_img, media_dir)
    
    return GenerationOutput(
        story=story,
        character_description=character_desc,
        background_description=background_desc,
        character_image_path=char_img,
        background_image_path=bg_img,
        combined_image_path=combined_img,
    )


