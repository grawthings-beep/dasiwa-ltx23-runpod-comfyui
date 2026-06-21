from pathlib import Path
import sys


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise SystemExit(f"Patch target not found:\n{old}")
    return text.replace(old, new, 1)


def patch_model_optimization(root: Path) -> None:
    path = root / "nodes" / "model_optimization_nodes.py"
    text = path.read_text(encoding="utf-8")

    if "leaving attention unpatched" not in text:
        text = replace_once(
            text,
            '''def get_sage_func(sage_attention, allow_compile=False):
    logging.info(f"Using sage attention mode: {sage_attention}")
    if sage_attention == "auto":
        from sageattention import sageattn
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
    elif "sageattn3" in sage_attention:
        from sageattn3 import sageattn3_blackwell
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
            q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
            out = sageattn3_blackwell(q, k, v, is_causal=is_causal, attn_mask=attn_mask, per_block_mean=(sage_attention == "sageattn3_per_block_mean"))
            return out.transpose(1, 2) if tensor_layout == "NHD" else out

    if not allow_compile:
''',
            '''def get_sage_func(sage_attention, allow_compile=False):
    logging.info(f"Using sage attention mode: {sage_attention}")
    try:
        if sage_attention == "auto":
            from sageattention import sageattn
            def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
        elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
            from sageattention import sageattn_qk_int8_pv_fp16_cuda
            def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
        elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
            from sageattention import sageattn_qk_int8_pv_fp16_triton
            def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
        elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
            from sageattention import sageattn_qk_int8_pv_fp8_cuda
            def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
        elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
            from sageattention import sageattn_qk_int8_pv_fp8_cuda
            def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
        elif "sageattn3" in sage_attention:
            from sageattn3 import sageattn3_blackwell
            def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
                q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
                out = sageattn3_blackwell(q, k, v, is_causal=is_causal, attn_mask=attn_mask, per_block_mean=(sage_attention == "sageattn3_per_block_mean"))
                return out.transpose(1, 2) if tensor_layout == "NHD" else out
        else:
            return None
    except Exception as exc:
        logging.warning(
            "SageAttention mode %s unavailable (%s); leaving attention unpatched.",
            sage_attention,
            exc,
        )
        return None

    if not allow_compile:
''',
        )

    if "new_attention is None" not in text:
        text = replace_once(
            text,
            '''        new_attention = get_sage_func(sage_attention, allow_compile=allow_compile)
        def attention_override_sage(func, *args, **kwargs):
''',
            '''        new_attention = get_sage_func(sage_attention, allow_compile=allow_compile)
        if new_attention is None:
            return model,

        def attention_override_sage(func, *args, **kwargs):
''',
        )

    if "SageAttention execution failed" not in text:
        text = replace_once(
            text,
            '''        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
''',
            '''        fallback_q, fallback_k, fallback_v = q, k, v
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
''',
        )
        text = replace_once(
            text,
            '''        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        if tensor_layout == "HND":
''',
            '''        try:
            out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        except Exception as exc:
            logging.warning("SageAttention execution failed (%s); falling back to PyTorch attention.", exc)
            return attention_pytorch(
                fallback_q,
                fallback_k,
                fallback_v,
                heads,
                mask=mask,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        if tensor_layout == "HND":
''',
        )

    path.write_text(text, encoding="utf-8")


def patch_fp16_accumulation(root: Path) -> None:
    """Make ModelPatchTorchSettings tolerate older PyTorch.

    torch.backends.cuda.matmul.allow_fp16_accumulation only exists on PyTorch
    2.7.1+. On the pinned 2.4 base it is absent, and KJNodes raises
    RuntimeError("Failed to set fp16 accumulation, ...") even when the feature
    is disabled, which aborts the whole prompt. Treat it as a warning and leave
    torch settings unchanged so the workflow keeps running, matching the
    SageAttention fallback philosophy.
    """
    path = root / "nodes" / "model_optimization_nodes.py"
    text = path.read_text(encoding="utf-8")

    if "leaving torch settings unchanged" in text:
        return

    replacement = (
        'logging.warning("ModelPatchTorchSettings: fp16 accumulation requires '
        'pytorch 2.7.1 or higher, which is unavailable; leaving torch settings '
        'unchanged.")'
    )
    targets = (
        'raise RuntimeError("Failed to set fp16 accumulation, this requires '
        'pytorch 2.7.1 or higher")',
        'raise RuntimeError("Failed to set fp16 accumulation, requires pytorch '
        'version 2.7.1 or higher")',
    )

    replaced = False
    for old in targets:
        if old in text:
            text = text.replace(old, replacement)
            replaced = True

    if not replaced:
        # Upstream may have reworded/removed the raise; do not fail the build.
        print(
            "WARNING: fp16 accumulation raise target not found in "
            "model_optimization_nodes.py; skipping fp16 fallback patch."
        )
        return

    path.write_text(text, encoding="utf-8")


def patch_ltxv(root: Path) -> None:
    path = root / "nodes" / "ltxv_nodes.py"
    text = path.read_text(encoding="utf-8")

    if "leaving LTX2 Memory Efficient Sage Attention Patch inactive" not in text:
        text = replace_once(
            text,
            '''        if _cuda_archs is None:
            raise RuntimeError("sageattention is not new enough version or could not determine CUDA architecture, cannot apply LTX2 Memory Efficient Sage Attention Patch.")
''',
            '''        if _cuda_archs is None:
            logging.warning("SageAttention is unavailable or unsupported; leaving LTX2 Memory Efficient Sage Attention Patch inactive.")
            return io.NodeOutput(model)
''',
        )

    if "leaving WanVideo Memory Efficient Sage Attention Patch inactive" not in text:
        text = replace_once(
            text,
            '''        if _cuda_archs is None:
            raise RuntimeError("sageattention is not new enough version or could not determine CUDA architecture, cannot apply WanVideo Memory Efficient Sage Attention Patch.")
''',
            '''        if _cuda_archs is None:
            logging.warning("SageAttention is unavailable or unsupported; leaving WanVideo Memory Efficient Sage Attention Patch inactive.")
            return io.NodeOutput(model)
''',
        )
        text = replace_once(
            text,
            '''        if _wan_apply_rope is None:
            raise RuntimeError("Could not import apply_rope from comfy.ldm.flux.math, cannot apply WanVideo Memory Efficient Sage Attention Patch.")
''',
            '''        if _wan_apply_rope is None:
            logging.warning("ComfyUI apply_rope is unavailable; leaving WanVideo Memory Efficient Sage Attention Patch inactive.")
            return io.NodeOutput(model)
''',
        )

    path.write_text(text, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_kjnodes_sage_fallback.py /path/to/ComfyUI-KJNodes")
    root = Path(sys.argv[1])
    patch_model_optimization(root)
    patch_fp16_accumulation(root)
    patch_ltxv(root)


if __name__ == "__main__":
    main()
