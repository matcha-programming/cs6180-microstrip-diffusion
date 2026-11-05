"""Protect scheduler submission from duplicate jobs after partial/unknown failures."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import campaign


class SubmissionTests(unittest.TestCase):
    def test_existing_journal_refuses_to_submit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(campaign, "ROOT", Path(tmp)), patch.object(campaign, "run") as run:
            (Path(tmp) / "submission.json").write_text('{}')
            with self.assertRaisesRegex(RuntimeError, "journal"):
                campaign.submit()
            run.assert_not_called()

    def test_unknown_sbatch_outcome_leaves_a_journal_for_inspection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(campaign, "ROOT", Path(tmp)), patch.object(campaign, "preflight", return_value={}), patch.object(campaign, "run", side_effect=subprocess.CalledProcessError(1, ["sbatch"])):
            with self.assertRaises(subprocess.CalledProcessError):
                campaign.submit()
            self.assertEqual(json.loads((Path(tmp) / "submission.json").read_text())["state"], "submitting")

    def test_dependency_failure_retains_submitted_array_id(self):
        reply = subprocess.CompletedProcess(["sbatch"], 0, stdout="12345\n")
        with tempfile.TemporaryDirectory() as tmp, patch.object(campaign, "ROOT", Path(tmp)), patch.object(campaign, "preflight", return_value={}), patch.object(campaign, "run", side_effect=[reply, subprocess.CalledProcessError(1, ["sbatch"])]):
            with self.assertRaises(subprocess.CalledProcessError):
                campaign.submit()
            record = json.loads((Path(tmp) / "submission.json").read_text())
            self.assertEqual(record["state"], "array_submitted")
            self.assertEqual(record["array_job_id"], "12345")


if __name__ == "__main__":
    unittest.main()
