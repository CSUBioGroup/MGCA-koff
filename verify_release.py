#!/usr/bin/env python3
"""Verify the integrity and publication hygiene of this MGCA release."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL_CASE_ID = "3154ac214828eafe59655160112af7c34eaf8be4cf739ed0ed21f2eb460d2b30"
STALE_CASE_IDS = ("69590ecd8276", "f3a43680dda9")
TEXT_SUFFIXES = {
    ".py", ".sh", ".ps1", ".json", ".jsonl", ".csv", ".tsv", ".md",
    ".txt", ".yml", ".yaml", ".cff", ".html", ".xml", ".toml", ".ini",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".npz", ".log", ".lock", ".zip"}
REQUIRED = (
    "LICENSE", "NOTICE", "CITATION.cff", ".zenodo.json", "README.md",
    "DATA_LICENSES.md", ".gitattributes", ".github/workflows/release-integrity.yml",
    "requirements.txt", "data", "model", "experiments/final_v10",
    "results/final_v10", "baselines", "case_study/code", "case_study/results",
    "inference_service", "MANIFEST.json", "SHA256SUMS",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    errors: list[str] = []
    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            fail(errors, f"missing required path: {rel}")

    files = [
        p for p in ROOT.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(ROOT).parts
    ]
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        low = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(errors, f"forbidden artifact: {rel}")
        if low == ".complete" or low.endswith(".complete"):
            fail(errors, f"transient completion marker: {rel}")
        if low == "train_predictions.csv" or low.startswith("attempt_"):
            fail(errors, f"excluded transient/training artifact: {rel}")

    searchable = []
    private_patterns = (
        re.compile(r"[A-Za-z]:[\\/]Users[\\/]NK", re.I),
        re.compile(r"E:[\\/]Desktop[\\/]Bio", re.I),
    )
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT).as_posix()
        searchable.append((rel, text))
        if any(p.search(text) for p in private_patterns):
            fail(errors, f"private absolute path remains: {rel}")

    # Historical helper modules remain in case_study/code because the final
    # wrapper imports their reusable functions while overriding their frozen
    # IDs. Only published case outputs must be free of superseded identities.
    case_text = "\n".join(
        text for rel, text in searchable if rel.startswith("case_study/results/")
    )
    if FINAL_CASE_ID not in case_text:
        fail(errors, "authoritative final case configuration ID is absent")
    for stale in STALE_CASE_IDS:
        if stale in case_text:
            fail(errors, f"stale case-study configuration ID remains: {stale}")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8", errors="replace") if (ROOT / "LICENSE").exists() else ""
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        fail(errors, "LICENSE is not recognizable as Apache-2.0")

    sums = ROOT / "SHA256SUMS"
    if sums.exists():
        listed: set[str] = set()
        for line_no, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                expected, rel = line.split("  ", 1)
            except ValueError:
                fail(errors, f"malformed SHA256SUMS line {line_no}")
                continue
            if rel in listed:
                fail(errors, f"duplicate SHA256SUMS entry: {rel}")
                continue
            listed.add(rel)
            target = ROOT / Path(rel)
            if not target.is_file():
                fail(errors, f"hash target missing: {rel}")
            elif sha256(target) != expected:
                fail(errors, f"hash mismatch: {rel}")
        expected_files = {
            p.relative_to(ROOT).as_posix() for p in files
            if p.name != "SHA256SUMS"
        }
        for rel in sorted(expected_files - listed):
            fail(errors, f"file absent from SHA256SUMS: {rel}")
        for rel in sorted(listed - expected_files):
            fail(errors, f"SHA256SUMS entry has no release file: {rel}")

    manifest_path = ROOT / "MANIFEST.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != "1.0.0":
                fail(errors, "MANIFEST release version must be 1.0.0")
            if manifest.get("scientific_identity", {}).get("final_case_config_id") != FINAL_CASE_ID:
                fail(errors, "MANIFEST final case configuration ID mismatch")
            if manifest.get("checkpoints_included") is not False:
                fail(errors, "MANIFEST must state checkpoints_included=false")
            for item in manifest.get("inventory", []):
                path = ROOT / item["path"]
                current = [p for p in path.rglob("*") if p.is_file()]
                current_bytes = sum(p.stat().st_size for p in current)
                if len(current) != item.get("files") or current_bytes != item.get("bytes"):
                    fail(errors, f"MANIFEST inventory mismatch: {item['path']}")
        except Exception as exc:
            fail(errors, f"invalid MANIFEST.json: {exc}")

    zenodo_path = ROOT / ".zenodo.json"
    if zenodo_path.exists():
        try:
            if json.loads(zenodo_path.read_text(encoding="utf-8")).get("version") != "1.0.0":
                fail(errors, ".zenodo.json release version must be 1.0.0")
        except Exception as exc:
            fail(errors, f"invalid .zenodo.json: {exc}")

    citation_path = ROOT / "CITATION.cff"
    if citation_path.exists():
        citation = citation_path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r'^version:\s*["\']?1\.0\.0["\']?\s*$', citation, re.M):
            fail(errors, "CITATION.cff release version must be 1.0.0")

    if errors:
        print("RELEASE VERIFICATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    size = sum(p.stat().st_size for p in files)
    print(f"Release verification passed: {len(files)} files, {size / 1024**2:.2f} MiB")
    print(f"Final case configuration: {FINAL_CASE_ID}")
    print("Checkpoint binaries: absent (as required)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
