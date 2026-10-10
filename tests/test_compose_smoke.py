"""The deployment smoke check accepts the viewer links the server generates."""

import pathlib
import runpy

import pytest
from ml4paleo_server.viewer import neuroglancer_link

SMOKE = runpy.run_path(
    str(pathlib.Path(__file__).resolve().parents[1] / ".github/scripts/check_upload.py")
)


@pytest.mark.parametrize("query", ["", "?v=obj1", "?v=next"])
def test_smoke_accepts_bundled_viewer_links_with_cache_versions(query):
    assert SMOKE["is_neuroglancer_link"](f"/neuroglancer/{query}#!%7B%7D")


def test_smoke_accepts_the_servers_current_viewer_link():
    link = neuroglancer_link("https://example.org", "/image/zarr/", {})
    assert SMOKE["is_neuroglancer_link"](link)


@pytest.mark.parametrize(
    "link",
    [
        None,
        "",
        "/neuroglancer/",
        "/neuroglancer/?v=obj1",
        "/neuroglancer/?v=obj1#not-a-state",
        "/another-viewer/#!{}",
        "//other.example/neuroglancer/#!{}",
        "https://other.example/neuroglancer/#!{}",
    ],
)
def test_smoke_still_rejects_missing_or_wrong_viewer_links(link):
    assert not SMOKE["is_neuroglancer_link"](link)
