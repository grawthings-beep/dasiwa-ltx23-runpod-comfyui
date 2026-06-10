import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_workflows
import generate_workflows


class WorkflowToolingTests(unittest.TestCase):
    def test_manifest_is_pinned_and_covers_workflows(self):
        _, cnr_ids, provided_types = audit_workflows.load_manifest(
            ROOT / "custom_nodes.lock.json"
        )
        errors = []
        for workflow in sorted((ROOT / "workflows").glob("*.json")):
            errors.extend(
                audit_workflows.audit_workflow(workflow, cnr_ids, provided_types)
            )
        self.assertEqual(errors, [])

    def test_generated_workflow_is_current(self):
        self.assertEqual(
            generate_workflows.OUTPUT.read_text(encoding="utf-8"),
            generate_workflows.render(),
        )

    def test_runpod_workflow_uses_installed_models_and_safe_defaults(self):
        workflow = json.loads(
            generate_workflows.OUTPUT.read_text(encoding="utf-8")
        )
        nodes = list(audit_workflows.workflow_nodes(workflow))
        by_type = {}
        for node in nodes:
            by_type.setdefault(node["type"], []).append(node)

        self.assertEqual(
            by_type["UNETLoader"][0]["widgets_values"][0],
            generate_workflows.DEFAULT_UNET,
        )
        self.assertEqual(
            by_type["DualCLIPLoader"][0]["widgets_values"][0],
            generate_workflows.DEFAULT_TEXT_ENCODER,
        )
        self.assertEqual(by_type["DualCLIPLoader"][0]["widgets_values"][3], "cpu")

        for node_type in generate_workflows.DISABLED_NODE_TYPES:
            for node in by_type[node_type]:
                self.assertEqual(node["mode"], 4)

        vae_names = {
            node["widgets_values"][0]
            for node_type in ("VAELoader", "VAELoaderKJ")
            for node in by_type[node_type]
        }
        self.assertTrue(all(name.startswith("LTX/") for name in vae_names))


if __name__ == "__main__":
    unittest.main()
