from pathlib import Path
import sys


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise SystemExit(f"Patch target not found:\n{old}")
    return text.replace(old, new, 1)


def patch_scaler(root: Path) -> None:
    path = root / "nodes" / "scaling_nodes.py"
    text = path.read_text(encoding="utf-8")

    if "def _normalize_bool" in text:
        return

    text = replace_once(
        text,
        '                "custom_aspect_width": ("INT", {"default": 16, "min": 1, "max": 8192, "description": "Used only when scale_from_image is USE ASPECT BELOW and aspect preset is CUSTOM. Ratio width, not final pixels."}),',
        '                "custom_aspect_width": ("STRING", {"default": "16", "description": "Used only when scale_from_image is USE ASPECT BELOW and aspect preset is CUSTOM. Ratio width, not final pixels."}),',
    )
    text = replace_once(
        text,
        '                "custom_aspect_height": ("INT", {"default": 9, "min": 1, "max": 8192, "description": "Used only when scale_from_image is USE ASPECT BELOW and aspect preset is CUSTOM. Ratio height, not final pixels."}),',
        '                "custom_aspect_height": ("STRING", {"default": "9", "description": "Used only when scale_from_image is USE ASPECT BELOW and aspect preset is CUSTOM. Ratio height, not final pixels."}),',
    )
    text = replace_once(
        text,
        '                "custom_divisor": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1, "description": "Custom pixel boundary snapping."}),',
        '                "custom_divisor": ("STRING", {"default": "8", "description": "Custom pixel boundary snapping."}),',
    )

    text = replace_once(
        text,
        '    CATEGORY = "DaSiWa/Scaling"\n\n    def calculate(',
        '''    CATEGORY = "DaSiWa/Scaling"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def validate_inputs(self, *args, **kwargs):
        return True

    @staticmethod
    def _normalize_bool(value, default=False):
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", ""}:
                return False
            return default
        return bool(value)

    @staticmethod
    def _normalize_int(value, default):
        try:
            value = int(value)
        except Exception:
            return default
        return value if value >= 1 else default

    def calculate(''',
    )

    text = replace_once(
        text,
        '''        aspect_preset = aspect_preset if aspect_preset is not None else aspect_preset_when_not_image
        swap_aspect = swap_aspect if swap_aspect is not None else swap_aspect_when_not_image
        manual_aspect_width = manual_aspect_width if manual_aspect_width is not None else custom_aspect_width
        manual_aspect_height = manual_aspect_height if manual_aspect_height is not None else custom_aspect_height

        # 1. GET SOURCE DIMENSIONS (From Image or Manual)
''',
        '''        aspect_preset = aspect_preset if aspect_preset is not None else aspect_preset_when_not_image
        swap_aspect = swap_aspect if swap_aspect is not None else swap_aspect_when_not_image
        manual_aspect_width = manual_aspect_width if manual_aspect_width is not None else custom_aspect_width
        manual_aspect_height = manual_aspect_height if manual_aspect_height is not None else custom_aspect_height

        if resolution_preset in {"Use Precision Presets", "Use Resolution Presets"}:
            legacy_method = resolution_preset
            legacy_precision = no_scale
            legacy_resolution = scale_from_image
            legacy_no_scale = aspect_preset_when_not_image
            legacy_scale_from_image = swap_aspect_when_not_image
            legacy_aspect = custom_aspect_width
            legacy_swap = custom_aspect_height
            legacy_width = mode
            legacy_height = custom_divisor

            resolution_preset = legacy_resolution if legacy_method == "Use Resolution Presets" else legacy_precision
            if resolution_preset not in self.PRESETS:
                resolution_preset = "0.83 MP - HD"
            no_scale = self._normalize_bool(legacy_no_scale, False)
            scale_from_image = self._normalize_bool(legacy_scale_from_image, False)
            aspect_preset = legacy_aspect if legacy_aspect in self.ASPECT_PRESETS else "9:16 - Social"
            swap_aspect = self._normalize_bool(legacy_swap, False)
            manual_aspect_width = self._normalize_int(legacy_width, 16)
            manual_aspect_height = self._normalize_int(legacy_height, 9)
            mode = "WAN/LTX (Div32)"
            custom_divisor = 8

        no_scale = self._normalize_bool(no_scale, False)
        scale_from_image = self._normalize_bool(scale_from_image, True)
        swap_aspect = self._normalize_bool(swap_aspect, False)
        manual_aspect_width = self._normalize_int(manual_aspect_width, 16)
        manual_aspect_height = self._normalize_int(manual_aspect_height, 9)
        custom_divisor = self._normalize_int(custom_divisor, 8)
        if resolution_preset not in self.PRESETS:
            resolution_preset = "0.52 MP - SD"
        if aspect_preset not in self.ASPECT_PRESETS:
            aspect_preset = "9:16 - Social"
        if mode not in {"Standard", "WAN/LTX (Div32)", "LTX 2-Stage (Div64)", "CUSTOM"}:
            mode = "WAN/LTX (Div32)"

        # 1. GET SOURCE DIMENSIONS (From Image or Manual)
''',
    )
    text = replace_once(
        text,
        "            d = max(1, int(custom_divisor))",
        "            d = max(1, custom_divisor)",
    )

    path.write_text(text, encoding="utf-8")


def patch_js(root: Path) -> None:
    path = root / "js" / "dasiwa_ui.js"
    text = path.read_text(encoding="utf-8")

    if "migrateLegacyScalerWidgets" in text:
        return

    text = replace_once(
        text,
        'const SCALER_NODE = "DaSiWa_ResolutionScaleCalculator";\n',
        '''const SCALER_NODE = "DaSiWa_ResolutionScaleCalculator";

function migrateLegacyScalerWidgets(node) {
    const widgets = node.widgets;
    if (!widgets || widgets.length < 9) return;

    const values = widgets.map(w => w.value);
    const method = values[0];
    if (method !== "Use Precision Presets" && method !== "Use Resolution Presets") return;

    const nextValues = [
        method === "Use Resolution Presets" ? values[2] : values[1],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9] ?? "WAN/LTX (Div32)",
        values[10] ?? 8,
    ];

    nextValues.forEach((value, index) => {
        if (widgets[index]) widgets[index].value = value;
    });
}
''',
    )

    text = replace_once(
        text,
        '''            return r;
        };
    }
});''',
        '''            if (nodeData.name === SCALER_NODE) {
                migrateLegacyScalerWidgets(this);
            }
            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
            if (nodeData.name === SCALER_NODE) {
                migrateLegacyScalerWidgets(this);
            }
            return r;
        };
    }
});''',
    )

    path.write_text(text, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_dasiwa_scaler_legacy.py /path/to/ComfyUI-DaSiWa-Nodes")
    root = Path(sys.argv[1])
    patch_scaler(root)
    patch_js(root)


if __name__ == "__main__":
    main()
