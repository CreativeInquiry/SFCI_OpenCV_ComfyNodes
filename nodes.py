"""
nodes.py


MaskToTracks

Turn any ComfyUI MASK into structured TRACKS data (point / box / contour / area
per blob per frame). No SAM3, no tracker — just traces whatever mask you give
it. Works with masks from any source: another model, a threshold, hand-painted.

It outputs the same `TRACKS` type used by ComfyUI-EasyTrack, so you can feed it
into EasyTrack's Tracks Export / Tracks Preview nodes if you have them — or
write your own consumer. (The `tracks.py` data model here is a copy of
EasyTrack's; keep the two in sync if you change the schema.)
"""

from __future__ import annotations

import numpy as np

from .tracks import (
    Tracks, FrameDet,
    mask_to_rle, bbox_from_mask, centroid_from_mask, mask_to_contours,
)


class MaskToTracks:
    """
    Any ComfyUI MASK (B,H,W, values 0..1) -> TRACKS, by tracing it.
    Each batch slot is treated as a frame.

    Identity note: a plain mask has no tracker, so in separate_objects mode the
    per-blob IDs are assigned per frame by size (largest = 0); they are NOT
    linked across frames. Use single-object mode for one evolving shape, or run a
    real tracker (e.g. SAM3) if you need identity over time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"masks": ("MASK", {"tooltip": "Any mask batch (B,H,W). Each slot is one frame."})},
            "optional": {
                "label": ("STRING", {"default": "mask", "tooltip": "Name for the traced objects."}),
                "separate_objects": ("BOOLEAN", {"default": True, "tooltip": "ON: split disconnected blobs into separate objects. OFF: treat the whole mask as one object."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Cutoff for turning the 0..1 mask into black-and-white."}),
                "min_area": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "display": "slider", "tooltip": "Drop blobs SMALLER than this fraction of the image area. 0 = no minimum. e.g. 0.001 removes specks."}),
                "max_area": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "display": "slider", "tooltip": "Drop blobs LARGER than this fraction of the image area. 1 = no maximum. e.g. 0.9 removes whole-frame blobs."}),
                "store_contour": ("BOOLEAN", {"default": True, "tooltip": "Save the traced outline (polygon) of each blob."}),
                "store_mask_rle": ("BOOLEAN", {"default": True, "tooltip": "Save the exact pixel mask (lossless). Turn OFF for smaller files."}),
                "contour_simplify": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 0.05, "step": 0.001, "tooltip": "Outline detail vs file size. 0 = keep every point; higher = fewer, smoother."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0, "tooltip": "Frames per second, stored for reference."}),
            },
        }

    RETURN_TYPES = ("TRACKS",)
    RETURN_NAMES = ("tracks",)
    OUTPUT_TOOLTIPS = ("Structured tracking data: per object, per frame point/box/contour/area.",)
    FUNCTION = "convert"
    CATEGORY = "MaskToTracks"
    DESCRIPTION = ("Trace any mask into usable data. Splits the mask into blobs, and for each "
                   "one works out the center point, bounding box, contour outline, and area. "
                   "Outputs the TRACKS type (compatible with ComfyUI-EasyTrack's export/preview).")

    def convert(self, masks, label="mask", separate_objects=True, threshold=0.5,
                min_area=0.0, max_area=1.0, store_contour=True, store_mask_rle=True,
                contour_simplify=0.002, fps=24.0):
        import cv2
        arr = masks.detach().cpu().numpy()
        if arr.ndim == 2:           # a single (H,W) mask -> one frame
            arr = arr[None, ...]
        B, H, W = arr.shape[0], arr.shape[1], arr.shape[2]
        image_area = float(max(H * W, 1))
        lo, hi = min_area * image_area, max_area * image_area   # blob size window
        tracks = Tracks(height=H, width=W, num_frames=B, fps=float(fps))

        for b in range(B):
            binary = (arr[b] > threshold).astype(np.uint8)
            if binary.sum() == 0:
                continue

            # split into blobs, keep only those whose size is in the [min,max] window
            n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            in_range = [i for i in range(1, n)                    # skip background (0)
                        if lo <= int(stats[i, cv2.CC_STAT_AREA]) <= hi]
            if not in_range:
                continue

            if separate_objects:
                # each surviving blob becomes its own object; largest -> id 0
                in_range.sort(key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
                for oid, i in enumerate(in_range):
                    cm = (labels == i).astype(np.uint8)
                    self._add(tracks, oid, b, cm, label,
                              store_contour, store_mask_rle, contour_simplify)
            else:
                # one object: union of the surviving blobs (specks already dropped)
                cm = np.zeros_like(binary)
                for i in in_range:
                    cm[labels == i] = 1
                self._add(tracks, 0, b, cm, label,
                          store_contour, store_mask_rle, contour_simplify)

        print(f"[MaskToTracks] -> {tracks!r}")
        return (tracks,)

    @staticmethod
    def _add(tracks, oid, frame, m, label, store_contour, store_mask_rle, contour_simplify):
        bbox = bbox_from_mask(m)
        if bbox is None:
            return
        tracks.add(oid, frame, FrameDet(
            bbox=bbox,
            point=centroid_from_mask(m),
            contour=(mask_to_contours(m, contour_simplify) if store_contour else None),
            area=int(m.sum()),
            score=1.0,
            visible=True,
            mask_rle=(mask_to_rle(m) if store_mask_rle else None),
        ), label=label, score=1.0)
