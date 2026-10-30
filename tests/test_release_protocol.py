"""CPU-only checks for deployment inputs, protocol guards and official metrics."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import evaluate
import predict
from iterblob_demo.official_structure_metrics import parse_tmscore_output, run_tmscore
from iterblob_demo.tier1_primitive_geometry import primitive_hinge_pivot, primitive_local_basis
from iterblob_demo.tier1_primitive_selection import select_primitive_from_pae
from iterblob_demo.tier1_residue_pair_attention import (
    build_residue_pair_tokens, create_pair_attention_pose_head,
)


# Existing project fixture from tests/test_official_structure_metrics.py.
TMSCORE_OUTPUT = """
Structure1: predicted.pdb    Length=  250
Structure2: native.pdb    Length=  166 (by which all scores are normalized)
Number of residues in common=  166
RMSD of  the common residues=   18.378

TM-score    = 0.1679  (d0= 4.80)
MaxSub-score= 0.0774  (d0= 3.50)
GDT-TS-score= 0.1265 %(d<1)=0.0783 %(d<2)=0.0783 %(d<4)=0.0964 %(d<8)=0.2530
GDT-HA-score= 0.0798 %(d<0.5)=0.0663 %(d<1)=0.0783 %(d<2)=0.0783 %(d<4)=0.0964
"""


class ReleaseProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.csv"
        self.partition_manifest = self.root / "partitions.csv"
        self.row = {"target": "synthetic", "split": "test", "cache_npz": "predicted.npz"}
        self.partition_row = {
            "target": "synthetic", "split": "test", "partition_npz": "partition.npz",
            "provider": "synthetic_predicted_partition", "input_source": "prediction_only",
        }
        predict.write_csv(self.manifest, [self.row])
        predict.write_csv(self.partition_manifest, [self.partition_row])
        self.prediction = {
            "sequence": np.asarray("AAAAAA"),
            "coordinates": np.asarray([
                [0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.2, 0.3],
                [8.0, 0.1, 0.2], [9.0, 0.4, 0.1], [8.2, 1.3, 0.4],
            ], dtype=np.float32),
            "residue_mask": np.ones(6, dtype=bool),
            "plddt": np.linspace(70.0, 95.0, 6, dtype=np.float32),
            "pae": 1.0 + np.arange(36, dtype=np.float32).reshape(6, 6) / 10.0,
        }
        self.partition = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        np.savez(self.root / "partition.npz", partition_ids=self.partition)
        self.write_cache()
        torch.manual_seed(17)
        self.model = create_pair_attention_pose_head(hidden_dim=32, latent_count=4)
        with torch.no_grad():
            self.model.pose_head[-1].weight.fill_(1e-3)
            self.model.pose_head[-1].bias.copy_(torch.tensor([.01, -.02, .03, -.04, .05, -.06]))
        self.model.eval()

    def write_cache(self, **extra):
        np.savez(self.root / "predicted.npz", **self.prediction, **extra)

    def loaded_tokens(self):
        values = predict.load_predicted_target(
            self.row, self.partition_row, self.manifest, self.partition_manifest,
        )
        coordinates, mask, partition = values["coordinates"], values["residue_mask"], values["partition_ids"]
        primitive, _, _ = select_primitive_from_pae(partition, mask, values["pae"])
        _, basis = primitive_local_basis(coordinates, mask, partition, primitive)
        pivot = primitive_hinge_pivot(coordinates, mask, partition, primitive)
        tokens = build_residue_pair_tokens(
            coordinates, mask, partition, values["plddt"], values["pae"],
            primitive_id=primitive, pivot=pivot, basis=basis,
        )
        return values, tokens

    def test_inference_succeeds_without_native_coordinates_or_domain_ids(self):
        with np.load(self.root / "predicted.npz", allow_pickle=False) as arrays:
            self.assertNotIn("native_coordinates", arrays.files)
            self.assertNotIn("domain_ids", arrays.files)
        values, tokens = self.loaded_tokens()
        self.assertNotIn("native_coordinates", values)
        self.assertNotIn("domain_ids", values)
        self.assertEqual(tokens.shape, (9, 17))
        with torch.inference_mode():
            pose = self.model(torch.from_numpy(tokens))["pose"]
        self.assertTrue(torch.isfinite(pose).all())
        self.assertGreater(float(pose.abs().sum()), 0.0)

        checkpoint = self.root / "checkpoint.pt"
        torch.save({"state_dict": self.model.state_dict(), "head_type": "ours_pair_attention",
                    "model_config": {"hidden_dim": 32, "latent_count": 4, "attention_heads": 4}}, checkpoint)
        args = argparse.Namespace(
            manifest=self.manifest, partition_manifest=self.partition_manifest,
            checkpoint=checkpoint, output_dir=self.root / "predictions",
            device="cpu", split="test", limit_targets=None, write_pdb=True, threads=1,
        )
        with patch.object(predict, "parse_args", return_value=args), redirect_stdout(io.StringIO()):
            self.assertEqual(predict.main(), 0)
        saved = predict.read_csv(args.output_dir / "predicted_actions.csv")
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["action_status"], "pose_head_applied")
        self.assertTrue((args.output_dir / "pdb" / "synthetic_pairrefine_mapped_ca.pdb").is_file())

    def test_adding_or_changing_native_data_does_not_change_tokens_or_pose(self):
        _, reference = self.loaded_tokens()
        with torch.inference_mode():
            reference_pose = self.model(torch.from_numpy(reference))["pose"]
        self.assertGreater(float(reference_pose.abs().sum()), 0.0)
        for native in (self.prediction["coordinates"].copy(), np.full((6, 3), np.nan, dtype=np.float32)):
            with self.subTest(native_is_finite=bool(np.isfinite(native).all())):
                self.write_cache(native_coordinates=native, domain_ids=np.full(6, 999, dtype=np.int64))
                values, observed = self.loaded_tokens()
                self.assertNotIn("native_coordinates", values)
                np.testing.assert_array_equal(reference, observed)
                with torch.inference_mode():
                    observed_pose = self.model(torch.from_numpy(observed))["pose"]
                torch.testing.assert_close(reference_pose, observed_pose, rtol=0, atol=0)

    def evaluate_invalid_action(self, selected_primitive, **pose_updates):
        self.write_cache(native_coordinates=self.prediction["coordinates"].copy())
        action = {"target": "synthetic", "selected_primitive_id": selected_primitive,
                  **dict.fromkeys(predict.POSE_FIELDS, 0.0), **pose_updates}
        actions = self.root / "actions.csv"
        predict.write_csv(actions, [action])
        args = argparse.Namespace(
            saved_results=actions, manifest=self.manifest,
            partition_manifest=self.partition_manifest, binary=Path(__file__),
            output_dir=self.root / "evaluation", dataset_name="synthetic",
            bootstrap_samples=10, seed=17, limit_targets=None, target=None,
        )
        return args

    def test_evaluation_rejects_nonidentity_single_partition_actions(self):
        np.savez(self.root / "partition.npz", partition_ids=np.zeros(6, dtype=np.int64))
        args = self.evaluate_invalid_action(-1, translation_x_angstroms=0.1)
        with patch.object(evaluate, "parse_args", return_value=args), patch.object(evaluate, "run_tmscore") as scorer:
            with self.assertRaisesRegex(ValueError, "single-primitive fallback must be the identity"):
                evaluate.main()
            scorer.assert_not_called()

    def test_evaluation_rejects_saved_primitive_that_disagrees_with_pae(self):
        expected, _, _ = select_primitive_from_pae(
            self.partition, self.prediction["residue_mask"], self.prediction["pae"],
        )
        args = self.evaluate_invalid_action(1 - expected)
        with patch.object(evaluate, "parse_args", return_value=args), patch.object(evaluate, "run_tmscore") as scorer:
            with self.assertRaisesRegex(ValueError, "saved primitive disagrees with PAE"):
                evaluate.main()
            scorer.assert_not_called()

    def test_pdb_export_rejects_coordinate_field_overflow(self):
        for value in (-1000.0, 9999.9996):
            coordinates = np.asarray([[value, 0.0, 0.0]], dtype=np.float64)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "fixed-width PDB"):
                predict.write_ca_pdb(
                    self.root / "overflow.pdb", "A", coordinates,
                    np.asarray([90.0]), np.asarray([True]),
                )


class OfficialMetricContractTests(unittest.TestCase):
    def test_parser_keeps_native_normalization_and_common_residue_rmsd(self):
        result = parse_tmscore_output(TMSCORE_OUTPUT)
        self.assertEqual(result.structure1_length, 250)
        self.assertEqual(result.structure2_length, 166)
        self.assertEqual(result.common_residues, 166)
        self.assertNotEqual(result.structure1_length, result.structure2_length)
        self.assertAlmostEqual(result.common_residue_rmsd, 18.378)
        self.assertAlmostEqual(result.tm_score, 0.1679)
        self.assertEqual(result.raw_stdout, TMSCORE_OUTPUT)

    def test_wrapper_passes_native_as_structure2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predicted, native = root / "predicted.pdb", root / "native.pdb"
            predicted.touch()
            native.touch()
            binary = Path(__file__)
            completed = SimpleNamespace(returncode=0, stdout=TMSCORE_OUTPUT, stderr="")
            with patch("iterblob_demo.official_structure_metrics.subprocess.run", return_value=completed) as run:
                result = run_tmscore(predicted, native, binary)
            self.assertEqual(run.call_args.args[0], [str(binary.resolve()), str(predicted.resolve()), str(native.resolve())])
            self.assertEqual(result.structure2_length, 166)
            self.assertAlmostEqual(result.common_residue_rmsd, 18.378)

    def test_parser_rejects_incomplete_output(self):
        with self.assertRaisesRegex(ValueError, "Structure1"):
            parse_tmscore_output("TM-score = 1.0")


if __name__ == "__main__":
    unittest.main()
