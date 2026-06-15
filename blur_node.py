import cv2
import numpy as np
import torch


class GaussianBlurNode:
    """
    Applies a Gaussian blur to an image using OpenCV.
    Kernel size controls how blurry the image gets.
    Sigma controls the shape of the blur — set to 0 to auto-calculate.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":       ("IMAGE",),
                "kernel_size": ("INT",   {"default": 5,   "min": 1, "max": 101, "step": 2}),
                "sigma":       ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("blurred_image",)
    FUNCTION      = "apply_blur"
    CATEGORY      = "tutorials/opencv"

    def apply_blur(self, image, kernel_size, sigma):

        # Safety check: kernel size must be odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # ── ComfyUI tensor → OpenCV array ──────────────────────────────────
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # ── Apply the Gaussian blur ─────────────────────────────────────────
        blurred = cv2.GaussianBlur(
            img_cv,
            (kernel_size, kernel_size),
            sigma
        )

        # ── OpenCV array → ComfyUI tensor ──────────────────────────────────
        blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        blurred_float = blurred_rgb.astype(np.float32) / 255.0
        output = torch.from_numpy(blurred_float).unsqueeze(0)

        return (output,)


NODE_CLASS_MAPPINGS = {
    "GaussianBlur": GaussianBlurNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GaussianBlur": "▓▓▓▓▓▓ Gaussian Blur ▓▓▓▓▓▓",
}
