"""Local viewer setup reuses assets and only publishes complete builds."""

import os
import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def run_build(tmp_path, *, fail=False):
    binaries = tmp_path / "bin"
    binaries.mkdir(exist_ok=True)
    # Exercise the script without fetching upstream or installing dependencies.
    (binaries / "git").write_text("#!/bin/sh\nexit 0\n")
    (binaries / "npm").write_text(
        "#!/bin/sh\n"
        + (
            "exit 1\n"
            if fail
            else """
if [ "$1" = run ]; then
    mkdir -p dist/client
    printf 'client' > dist/client/main.js
    printf 'viewer' > dist/client/index.html
fi
"""
        )
    )
    for binary in binaries.iterdir():
        binary.chmod(0o755)
    output = tmp_path / "viewer"
    result = subprocess.run(
        ["bash", str(REPO / "deploy/build-neuroglancer.sh"), str(output)],
        env={
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "TMPDIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )
    return result, output


def test_local_build_publishes_assets_and_cleans_its_scratch(tmp_path):
    result, output = run_build(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (output / "index.html").read_text() == "viewer"
    assert (output / "main.js").read_text() == "client"
    assert (output / ".ml4paleo-build").read_text().strip()
    assert not list(tmp_path.glob("ml4paleo-neuroglancer.*"))


def test_local_start_reuses_an_existing_build_without_git_or_npm(tmp_path):
    result, output = run_build(tmp_path)
    assert result.returncode == 0, result.stderr
    (output / "index.html").write_text("already built")
    result, _ = run_build(tmp_path, fail=True)
    assert result.returncode == 0, result.stderr
    assert (output / "index.html").read_text() == "already built"


@pytest.mark.parametrize("revision", [None, "old revision"])
def test_local_start_rebuilds_an_unpatched_or_outdated_client(tmp_path, revision):
    output = tmp_path / "viewer"
    output.mkdir()
    (output / "index.html").write_text("unpatched")
    if revision is not None:
        (output / ".ml4paleo-build").write_text(revision)
    result, _ = run_build(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (output / "index.html").read_text() == "viewer"


def test_failed_build_does_not_look_ready_or_replace_existing_files(tmp_path):
    output = tmp_path / "viewer"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    result, _ = run_build(tmp_path, fail=True)
    assert result.returncode != 0
    assert not (output / "index.html").exists()
    assert (output / "keep.txt").read_text() == "keep"
    assert not list(tmp_path.glob("ml4paleo-neuroglancer.*"))


def test_failed_upgrade_preserves_the_previous_client(tmp_path):
    output = tmp_path / "viewer"
    output.mkdir()
    (output / "index.html").write_text("previous client")
    (output / ".ml4paleo-build").write_text("previous revision")
    result, _ = run_build(tmp_path, fail=True)
    assert result.returncode != 0
    assert (output / "index.html").read_text() == "previous client"
    assert (output / ".ml4paleo-build").read_text() == "previous revision"


def test_obj_patch_fixes_the_worker_reference_error(tmp_path):
    source = tmp_path / "src/util/array.ts"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class TypedArrayBuilder {\n"
        "  shrinkToFit() {\n"
        "    this.data = this.data.slice(0, length) as T;\n"
        "  }\n"
        "}\n"
    )
    result = subprocess.run(
        ["git", "apply", str(REPO / "deploy/neuroglancer-obj.patch")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "slice(0, this.length)" in source.read_text()


@pytest.mark.parametrize("pin", ["", "ARG NEUROGLANCER_COMMIT=invalid\n"])
def test_missing_pin_fails_before_a_checkout(tmp_path, pin):
    deploy = tmp_path / "repo/deploy"
    (deploy / "docker").mkdir(parents=True)
    shutil.copyfile(
        REPO / "deploy/build-neuroglancer.sh", deploy / "build-neuroglancer.sh"
    )
    (deploy / "docker/server.Dockerfile").write_text(pin)
    result = subprocess.run(
        ["bash", str(deploy / "build-neuroglancer.sh"), str(tmp_path / "viewer")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid Neuroglancer commit" in result.stderr
