"""Verify the public PreemptBench engine release without importing vLLM."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "PREEMPTBENCH_PROVENANCE.json"
EXPECTED_BASE = "2a69949bdadf0e8942b7a1619b229cb475beef20"
INTERNAL_CAPTURE = "799a0dc383b43791ed820c421ee3292d810b9635"
EXPECTED_PUBLIC_URL = "https://github.com/KevinHuangW/preemptbench-vllm"
EXPECTED_HARNESS_URL = "https://github.com/KevinHuangW/preemptbench"
EXPECTED_TREES = {
    "vllm": "ff66acc11d7f0f0a0eb23a543885a53244a7eea0",
    "docker": "9abe40dc1609ea40129c1528da2bcc8864356a1c",
}
EXCLUDED_PATHS = {
    "m1_traces/agent_calls.jsonl",
    "vllm/v1/core/probe_victim.py",
    "tests/v1/core/test_probe_victim_selection.py",
}


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )


class ReleaseContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    def test_upstream_base_is_ancestor(self) -> None:
        result = git("merge-base", "--is-ancestor", EXPECTED_BASE, "HEAD", check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_trees_are_exact(self) -> None:
        self.assertEqual(self.provenance["runtime_trees"], EXPECTED_TREES)
        for path, expected in EXPECTED_TREES.items():
            actual = git("rev-parse", f"HEAD:{path}").stdout.strip()
            self.assertEqual(actual, expected, path)

    def test_runtime_worktree_is_clean(self) -> None:
        status = git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "vllm",
            "docker",
        ).stdout
        self.assertEqual(status, "")

    def test_internal_history_and_pilot_files_are_absent(self) -> None:
        capture = git("cat-file", "-e", f"{INTERNAL_CAPTURE}^{{commit}}", check=False)
        self.assertNotEqual(capture.returncode, 0)
        tracked = set(git("ls-files").stdout.splitlines())
        self.assertTrue(EXCLUDED_PATHS.isdisjoint(tracked))

    def test_no_unrelated_history_is_reachable(self) -> None:
        unrelated = git("rev-list", "--all", "--not", "HEAD").stdout.splitlines()
        self.assertEqual(unrelated, [])

    def test_required_runtime_defaults_and_policy_keys_are_documented(self) -> None:
        expected_settings = {
            "PROBE_REACTIVE_OFFLOAD": "0",
            "PROBE_SHIELD_NEAR_COMPLETE_K": "0",
            "PROBE_MAX_PREEMPTS_PER_REQ": "4",
            "PROBE_DEFAULT_LIFO_SHIELD": "0",
        }
        self.assertEqual(
            self.provenance["headline_runtime_settings"], expected_settings
        )
        policies = {
            (entry["paper"], entry["harness"], entry["engine"])
            for entry in self.provenance["evaluated_policy_map"]
        }
        self.assertIn(("smallest_kv", "shortest_first", "shortest_first"), policies)
        self.assertIn(
            (
                "empirical_statistical_length_oracle",
                "oracle_remaining",
                "oracle",
            ),
            policies,
        )
        self.assertEqual(len(policies), 7)

    def test_public_repository_cross_links(self) -> None:
        self.assertEqual(self.provenance["repository"], "preemptbench-vllm")
        self.assertEqual(self.provenance["public_url"], EXPECTED_PUBLIC_URL)
        self.assertEqual(
            self.provenance["companion_harness_url"], EXPECTED_HARNESS_URL
        )
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(EXPECTED_HARNESS_URL, readme)

    def test_apache_license_is_retained(self) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0, January 2004", license_text)


if __name__ == "__main__":
    unittest.main()
