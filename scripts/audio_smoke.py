"""CPU Comfy API/mask integration + actual stream-copy test, no LTX weights."""
import importlib.util
import os
from pathlib import Path
import sys
from unittest import mock


def main():
    app = Path(os.environ.get("COMFYUI_APP", "/opt/ComfyUI"))
    sys.path.insert(0, str(app))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--cpu"]
    import torch
    from comfy_api.latest import io
    from comfy_extras.nodes_lt import LTXVConcatAVLatent

    path = app / "custom_nodes/DaSiWa-LTX-Audio"
    spec = importlib.util.spec_from_file_location("audio_pack_smoke", path / "__init__.py", submodule_search_locations=[str(path)])
    pack = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pack
    spec.loader.exec_module(pack)
    engine = pack.engine
    video = torch.arange(128, dtype=torch.float32).reshape(1, 128, 1, 1, 1)
    audio = {"samples": torch.zeros((1, 8, 10, 16)), "type": "audio"}
    calls = []

    def sample(noise, guider, sampler, sigmas, latent):
        v, a = latent["samples"].unbind()
        vm, am = latent["noise_mask"].unbind()
        assert torch.equal(v, video) and torch.count_nonzero(vm) == 0
        assert torch.all(am == 1)
        calls.append(float(a.mean()))
        # Deliberately corrupt the returned video: next pass MUST reattach source.
        result = LTXVConcatAVLatent.execute({"samples": v + 99, "noise_mask": vm},
            {"samples": a + 1, "noise_mask": am}).result[0]
        return io.NodeOutput(result, result)

    with mock.patch.object(engine.CFGGuider, "execute", return_value=io.NodeOutput(object())), \
         mock.patch.object(engine.BasicScheduler, "execute", return_value=io.NodeOutput(torch.tensor([1., 0.]))), \
         mock.patch.object(engine.SamplerCustomAdvanced, "execute", side_effect=sample):
        result = engine.sample_frozen(None, [], [], video, audio, 42, True)
        assert calls == [0., 1., 2.] and torch.all(result["samples"] == 3)
        assert torch.equal(video, torch.arange(128).reshape(video.shape))
    print("AUDIO MASK SMOKE: actual Comfy NestedTensor concat/split; video zero-mask each pass; sampler stub only, not GPU inference", flush=True)
    import unittest
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"), pattern="test_audio.py")
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
