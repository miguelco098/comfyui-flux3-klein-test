import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import flux3_rewrite as fr  # noqa: E402
import fetch_models as fm  # noqa: E402


def load(name):
    with open(os.path.join(ROOT, name)) as f:
        return json.load(f)


class RewriteTests(unittest.TestCase):
    def setUp(self):
        for k in list(os.environ):
            if k.startswith("FLUX3_"):
                del os.environ[k]

    def test_base_switches_models_keeps_sampler(self):
        wf = load("flux2_klein_base_t2i_api.json")
        res = fr.rewrite_workflow(wf, fr.options_from(None))
        out = res.workflow
        self.assertEqual(out["1"]["inputs"]["unet_name"], "flux3-image-klein-base-bf16.safetensors")
        self.assertEqual(out["2"]["inputs"]["clip_name"], "qwen3vl_4b_int8_convrot.safetensors")
        self.assertEqual(out["2"]["inputs"]["type"], "flux3")
        self.assertEqual(out["7"]["inputs"]["steps"], 30)
        self.assertEqual(out["7"]["inputs"]["cfg"], 3.5)
        # untouched
        self.assertEqual(out["3"], wf["3"])
        self.assertEqual(out["4"], wf["4"])
        self.assertEqual(out["6"], wf["6"])
        # input not mutated
        self.assertEqual(wf["1"]["inputs"]["unet_name"], "flux-2-klein-base-4b.safetensors")
        self.assertTrue(res.touched_model)

    def test_rewritten_base_equals_reference_flux3_workflow(self):
        res = fr.rewrite_workflow(load("flux2_klein_base_t2i_api.json"), fr.options_from(None))
        ref = load("flux3_klein_base_t2i_api.json")
        for nid in ("1", "2", "7"):
            self.assertEqual(res.workflow[nid]["inputs"], ref[nid]["inputs"], nid)

    def test_distilled_per_request(self):
        res = fr.rewrite_workflow(
            load("flux2_klein_base_t2i_api.json"), fr.options_from({"variant": "distilled"})
        )
        ks = res.workflow["7"]["inputs"]
        self.assertEqual(res.workflow["1"]["inputs"]["unet_name"], "flux3-image-klein-bf16.safetensors")
        self.assertEqual((ks["steps"], ks["cfg"]), (16, 1.0))

    def test_distilled_custom_steps_env(self):
        os.environ["FLUX3_VARIANT"] = "distilled"
        os.environ["FLUX3_STEPS"] = "8"
        res = fr.rewrite_workflow(load("flux2_klein_base_t2i_api.json"), fr.options_from(None))
        self.assertEqual(res.workflow["7"]["inputs"]["steps"], 8)

    def test_disabled(self):
        wf = load("flux2_klein_base_t2i_api.json")
        res = fr.rewrite_workflow(wf, fr.options_from({"enabled": False}))
        self.assertEqual(res.workflow, wf)
        self.assertEqual(res.changes, [])

    def test_flux3_workflow_untouched(self):
        wf = load("flux3_klein_distilled_t2i_api.json")
        res = fr.rewrite_workflow(wf, fr.options_from(None))
        self.assertEqual(res.workflow, wf)

    def test_wan_and_other_models_untouched(self):
        wf = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
            "3": {"class_type": "KSampler", "inputs": {"steps": 4, "cfg": 1.0}},
        }
        res = fr.rewrite_workflow(wf, fr.options_from(None))
        self.assertEqual(res.workflow, wf)

    def test_linked_inputs_not_overwritten(self):
        wf = load("flux2_klein_base_t2i_api.json")
        wf["7"]["inputs"]["steps"] = ["99", 0]
        res = fr.rewrite_workflow(wf, fr.options_from({"variant": "distilled"}))
        self.assertEqual(res.workflow["7"]["inputs"]["steps"], ["99", 0])
        self.assertEqual(res.workflow["7"]["inputs"]["cfg"], 1.0)

    def test_gguf_loader_becomes_unetloader(self):
        wf = {"1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux-2-klein-9b-Q8_0.gguf"}}}
        res = fr.rewrite_workflow(wf, fr.options_from(None))
        self.assertEqual(res.workflow["1"]["class_type"], "UNETLoader")
        self.assertEqual(res.workflow["1"]["inputs"]["weight_dtype"], "default")

    def test_bad_variant(self):
        with self.assertRaises(ValueError):
            fr.options_from({"variant": "turbo"})


class FetchPathTests(unittest.TestCase):
    def test_volume_layout_matches_flux2_endpoint(self):
        root = "/runpod-volume/models"
        self.assertEqual(
            fm.target_path(root, "diffusion_models/flux3-image-klein-base-bf16.safetensors"),
            "/runpod-volume/models/unet/flux3-image-klein-base-bf16.safetensors",
        )
        self.assertEqual(
            fm.target_path(root, "text_encoders/qwen3vl_4b_int8_convrot.safetensors"),
            "/runpod-volume/models/clip/qwen3vl_4b_int8_convrot.safetensors",
        )
        self.assertEqual(
            fm.target_path(root, "vae/flux2-vae.safetensors"),
            "/runpod-volume/models/vae/flux2-vae.safetensors",
        )

    def test_container_layout(self):
        self.assertEqual(
            fm.target_path("/comfyui/models", "diffusion_models/x.safetensors"),
            "/comfyui/models/diffusion_models/x.safetensors",
        )


if __name__ == "__main__":
    unittest.main()
