"""Save PSD — one layer per prompt/mask, each with a real editable layer mask.

The node is meant to sit at the end of a "prompt list -> SAM3 -> masks" graph:
every prompt produces one or more masks, and every mask becomes a layer that
holds a full copy of the source image plus a Photoshop layer mask named after
the prompt. Nothing is baked in: in Photoshop the mask can be painted on and
the hidden pixels are still there.

INPUT_IS_LIST is on, so the node collects everything the upstream graph
produced across its per-prompt executions.
"""

import json
import os

import numpy as np
import torch
from PIL import Image

import folder_paths

PSD_TOOLS_MIN = "1.11"

try:
    import psd_tools
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, PixelLayer
    from psd_tools.constants import ChannelID, Compression, TaggedBlockID
    from psd_tools.psd.layer_and_mask import (
        ChannelData,
        ChannelInfo,
        MaskData,
        MaskFlags,
    )

    # psd-tools renamed the writing API in 1.11.0: PixelLayer.frompil() took
    # `layer_name` and Group.new() took `parent` last. On <1.11 the calls below
    # would fail loudly (Group.new) or, worse, silently name every layer
    # "Layer", so refuse to load instead of writing a broken PSD.
    import inspect as _inspect

    if (
        not hasattr(Group, "new")
        or next(iter(_inspect.signature(Group.new).parameters)) != "parent"
    ):
        raise ImportError(
            f"psd-tools {getattr(psd_tools, '__version__', '?')} is too old, "
            f"need >= {PSD_TOOLS_MIN}"
        )

    PSD_TOOLS_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the environment
    PSDImage = None
    PSD_TOOLS_ERROR = exc


def _first(value, default=None):
    """INPUT_IS_LIST turns every widget value into a one-item list."""
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _iter_images(image_inputs):
    """-> list of HxWx3 float arrays"""
    out = []
    if image_inputs is None:
        return out
    items = image_inputs if isinstance(image_inputs, list) else [image_inputs]
    for t in items:
        if t is None:
            continue
        if isinstance(t, list):
            out.extend(_iter_images(t))
            continue
        t = t if t.dim() == 4 else t.unsqueeze(0)
        for i in range(t.shape[0]):
            out.append(t[i].detach().cpu().float().numpy())
    return out


def _iter_mask_groups(mask_inputs):
    """-> list of groups, each group is a list of HxW float arrays.

    One group per upstream execution (i.e. per prompt) when the graph fans out
    over a prompt list; a single group holding the whole batch otherwise.
    """
    groups = []
    if mask_inputs is None:
        return groups
    items = mask_inputs if isinstance(mask_inputs, list) else [mask_inputs]
    for t in items:
        if t is None:
            continue
        if isinstance(t, list):
            groups.extend(_iter_mask_groups(t))
            continue
        t = t.detach().cpu().float()
        if t.dim() == 2:
            t = t.unsqueeze(0)
        groups.append([t[i].numpy() for i in range(t.shape[0])])
    return groups


def _flatten_names(src):
    out = []
    if src is None:
        return out
    items = src if isinstance(src, list) else [src]
    for item in items:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            out.extend(str(x) for x in item)
        else:
            out.append(str(item))
    return [n for n in (s.strip() for s in out) if n]


def _collect_names(prompt_list, names):
    """`names` wins over `prompt_list` — the two are never concatenated.

    `prompt_list` holds one string per prompt and only lines up with the masks
    while every prompt owns exactly one of them. `names` is expected to carry
    one string per mask, which stays correct even when a prompt produced
    several detections or an upstream node flattened the per-prompt batches.
    Summing both sources (the old behaviour) doubled the list whenever both
    inputs were wired and pushed naming into the positional fallback.
    """
    per_mask = _flatten_names(names)
    if per_mask:
        return per_mask
    return _flatten_names(prompt_list)


def _to_pil_rgb(arr):
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def _to_pil_mask(arr, size):
    m = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    if m.size != size:
        m = m.resize(size, Image.BILINEAR)
    return m


def _set_layer_name(layer, name):
    """Name a layer so that non-latin prompts survive.

    A PSD stores the layer name twice: a legacy pascal string in the record
    (macroman, so `psd.save()` raises UnicodeEncodeError on anything Cyrillic)
    and a 'luni' tagged block with the real unicode name. Photoshop reads the
    unicode one, so the legacy field only needs to be encodable.
    """
    layer._record.name = name.encode("macroman", "replace").decode("macroman")
    layer._record.tagged_blocks.set_data(TaggedBlockID.UNICODE_LAYER_NAME, name)


def _attach_layer_mask(layer, mask_pil, crop_to_mask=False):
    """Attach a PIL "L" image to `layer` as an editable Photoshop layer mask.

    psd-tools has no high-level API for creating layer masks, so the raw
    records are built here: a USER_LAYER_MASK channel plus a MaskData record.
    Photoshop hides whatever the mask paints black; the layer pixels stay
    intact, so the mask can be repainted or removed at any time.
    """
    top, left = 0, 0
    right, bottom = mask_pil.width, mask_pil.height
    if crop_to_mask:
        bbox = mask_pil.getbbox()
        if bbox:
            left, top, right, bottom = bbox
            mask_pil = mask_pil.crop(bbox)

    channel = ChannelData(Compression.RLE)
    channel.set_data(mask_pil.tobytes(), mask_pil.width, mask_pil.height, 8)

    layer._record.mask_data = MaskData(
        top=top,
        left=left,
        bottom=bottom,
        right=right,
        # everything outside the mask rectangle is hidden
        background_color=0,
        flags=MaskFlags(),
    )
    layer._record.channel_info.append(
        ChannelInfo(id=ChannelID.USER_LAYER_MASK, length=len(channel.data) + 2)
    )
    layer._channels.append(channel)


class SavePSDLayers:
    CATEGORY = "image/psd"
    FUNCTION = "run"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("psd_path", "layer_count")
    DESCRIPTION = (
        "Writes a PSD where every mask becomes a layer: a full copy of the "
        "image with an editable Photoshop layer mask, named after the prompt "
        "it came from."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "masks": ("MASK",),
                "filename_prefix": ("STRING", {"default": "psd/ComfyUI"}),
                "include_original": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Add the untouched image as the bottom layer.",
                    },
                ),
                "only_first_visible": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Open the file with only the topmost masked layer visible instead of all of them stacked.",
                    },
                ),
                "invert_masks": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "advanced": True,
                        "tooltip": "Enable when the mask marks what should be hidden (e.g. the LoadImage mask).",
                    },
                ),
                "group_per_prompt": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "advanced": True,
                        "tooltip": "Put multiple detections of one prompt into a Photoshop group named after that prompt.",
                    },
                ),
                "crop_to_mask": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "advanced": True,
                        "tooltip": "Crop the layer mask channel to its bounding box. The layer itself keeps the full image; everything outside the mask rectangle stays hidden.",
                    },
                ),
            },
            "optional": {
                "prompt_list": ("LIST",),
                "names": ("STRING", {"forceInput": True}),
            },
        }

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _pair(groups, names):
        """-> list of (name, mask array)"""
        flat = [m for g in groups for m in g]

        if names and len(names) == len(groups):
            pairs = []
            for name, group in zip(names, groups):
                if len(group) == 1:
                    pairs.append((name, group[0]))
                else:
                    for i, m in enumerate(group, start=1):
                        pairs.append((f"{name} {i}", m))
            return pairs

        if names and len(names) == len(flat):
            return list(zip(names, flat))

        pairs = []
        for i, m in enumerate(flat):
            pairs.append((names[i] if names and i < len(names) else f"layer {i + 1}", m))
        return pairs

    @staticmethod
    def _group_index(groups, names):
        """Which prompt each flat mask belongs to (for group_per_prompt)."""
        idx = []
        for gi, group in enumerate(groups):
            label = names[gi] if names and gi < len(names) else f"prompt {gi + 1}"
            for _ in group:
                idx.append((gi, label))
        return idx

    # ------------------------------------------------------------------ run
    def run(
        self,
        image,
        masks,
        filename_prefix="psd/ComfyUI",
        include_original=True,
        only_first_visible=False,
        invert_masks=False,
        group_per_prompt=False,
        crop_to_mask=False,
        prompt_list=None,
        names=None,
    ):
        if PSDImage is None:
            raise RuntimeError(
                "psd-tools is required for the Save PSD node. Install it into the "
                f"ComfyUI python environment: pip install 'psd-tools>={PSD_TOOLS_MIN}'  "
                f"(import error: {PSD_TOOLS_ERROR})"
            )

        filename_prefix = _first(filename_prefix, "psd/ComfyUI")
        include_original = bool(_first(include_original, True))
        only_first_visible = bool(_first(only_first_visible, False))
        invert_masks = bool(_first(invert_masks, False))
        group_per_prompt = bool(_first(group_per_prompt, False))
        crop_to_mask = bool(_first(crop_to_mask, False))

        images = _iter_images(image)
        if not images:
            raise ValueError("Save PSD: no image on the input")

        groups = _iter_mask_groups(masks)
        flat_count = sum(len(g) for g in groups)
        if flat_count == 0:
            raise ValueError("Save PSD: no masks on the input")

        layer_names = _collect_names(prompt_list, names)
        pairs = self._pair(groups, layer_names)
        group_index = self._group_index(groups, layer_names)

        # Naming depends on how the masks arrived: one name per prompt only
        # works while the per-prompt grouping survives the graph. Say out loud
        # what was matched, so a silent mismatch (names shifted by one, tail
        # called "layer N") is visible in the log instead of only in the PSD.
        if layer_names and len(layer_names) not in (len(groups), flat_count):
            print(
                f"[Save PSD] ВНИМАНИЕ: имён {len(layer_names)}, а групп масок "
                f"{len(groups)} (кадров всего {flat_count}). Имена расставлены "
                f"по порядку и почти наверняка съехали. Если промт может дать "
                f"несколько находок (max_detections=-1), подайте в 'names' "
                f"список с именем на КАЖДУЮ маску."
            )
        else:
            print(
                f"[Save PSD] групп масок {len(groups)}, кадров {flat_count}, "
                f"имён {len(layer_names)} -> слоёв {len(pairs)}"
            )

        base = images[0]
        h, w = base.shape[:2]
        size = (w, h)

        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), w, h)
        )
        psd_name = f"{filename}_{counter:05}_.psd"
        psd_path = os.path.join(full_output_folder, psd_name)

        psd = PSDImage.new("RGB", size)

        if include_original:
            original = PixelLayer.frompil(
                _to_pil_rgb(base), psd, name="original", compression=Compression.RLE
            )
            psd.append(original)

        open_groups = {}
        # layers are appended bottom-up: walk the prompts backwards so that in
        # Photoshop the panel reads top-down in prompt order
        for i in range(len(pairs) - 1, -1, -1):
            name, mask = pairs[i]
            src = images[i] if len(images) == len(pairs) else base
            if src.shape[:2] != (h, w):
                src = np.asarray(
                    Image.fromarray(
                        np.clip(src * 255, 0, 255).astype(np.uint8)
                    ).resize(size, Image.BILINEAR),
                    dtype=np.float32,
                ) / 255.0

            m = 1.0 - mask if invert_masks else mask
            rgb = _to_pil_rgb(src)
            mask_pil = _to_pil_mask(m, size)

            parent = psd
            if group_per_prompt and i < len(group_index):
                gi, label = group_index[i]
                sizes = [len(g) for g in groups]
                if gi < len(sizes) and sizes[gi] > 1:
                    if gi not in open_groups:
                        grp = Group.new(psd, open_folder=True)
                        _set_layer_name(grp, label)
                        open_groups[gi] = grp
                    parent = open_groups[gi]

            # non-destructive: the layer keeps the full, untouched image; the
            # cut-out lives in a separate editable Photoshop layer mask
            layer = PixelLayer.frompil(rgb, psd, compression=Compression.RLE)
            _set_layer_name(layer, name)
            parent.append(layer)
            _attach_layer_mask(layer, mask_pil, crop_to_mask)
            if only_first_visible and i != 0:
                layer.visible = False

        psd.save(psd_path)

        # no preview file is written: the node hands the frontend a reference to
        # the saved PSD and the JS extension turns it into a browser download
        rel = os.path.join(subfolder, psd_name) if subfolder else psd_name
        return {
            "ui": {
                "psd": [
                    {
                        "filename": psd_name,
                        "subfolder": subfolder,
                        "type": "output",
                    }
                ],
                "text": [rel],
            },
            "result": (psd_path, len(pairs) + (1 if include_original else 0)),
        }


NODE_CLASS_MAPPINGS = {"SavePSDLayers": SavePSDLayers}
NODE_DISPLAY_NAME_MAPPINGS = {"SavePSDLayers": "Save PSD (masked layers)"}
