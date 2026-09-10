# SPDX-License-Identifier: MIT
"""Compatibility shims for the DaSiWa LTX 2.3 workflow.

The provided V39 workflow contains a ``ColorTransfer`` node that is not shipped
by any of the installed custom-node packs (it came from the workflow author's
local setup). In the workflow it is left bypassed, but ComfyUI still needs the
node *type* to be registered, otherwise it shows a "Missing Node Types" error
and the bypassed node's output cannot pass through to downstream nodes, which
breaks validation.

This module registers a faithful, dependency-free ``ColorTransfer`` that simply
returns ``image_target`` unchanged. That matches the bypassed behaviour exactly
(bypass = pass the primary image through), so the workflow loads cleanly and
runs. If you later install a real color-transfer pack, its registration can
replace this one.
"""


class ColorTransfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_target": ("IMAGE",),
                "method": (
                    ["reinhard_lab", "reinhard_rgb", "mean", "none"],
                    {"default": "reinhard_lab"},
                ),
                "mode": (
                    ["per_frame", "first_frame", "global"],
                    {"default": "per_frame"},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "image_ref": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "transfer"
    CATEGORY = "compat/DaSiWa"
    DESCRIPTION = (
        "Compatibility passthrough for the orphaned 'ColorTransfer' node used by "
        "the DaSiWa LTX 2.3 workflow. Returns image_target unchanged so the "
        "workflow loads and runs without the original (unavailable) node."
    )

    def transfer(
        self,
        image_target,
        method="reinhard_lab",
        mode="per_frame",
        strength=0.75,
        image_ref=None,
    ):
        # Passthrough: identical to the node being bypassed in the workflow.
        return (image_target,)


NODE_CLASS_MAPPINGS = {
    "ColorTransfer": ColorTransfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorTransfer": "Color Transfer (compat passthrough)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
