"""
A project's data in Neuroglancer: the link to its view, the endpoint that gives
it, and the policy that lets the web app frame Neuroglancer (and nobody else).
"""

import json
import re
import uuid
from urllib.parse import unquote

import pytest
from helpers import run_db, signup
from ml4paleo_server import artifacts
from ml4paleo_server.app import create_app
from ml4paleo_server.viewer import neuroglancer_link

PUBLIC = "https://ml4paleo.example.org"
IMAGE = "/api/projects/p/artifacts/i/zarr/"
LABELS = "/api/projects/p/labels/zarr/"
PREDICTION = "/api/projects/p/artifacts/q/zarr/"
SEGMENTATION = "/api/projects/p/artifacts/s/zarr/"
CLASSES = [(2, "#E8A33D"), (3, "#3d9be8")]
MESHES = "/api/projects/p/artifacts/m/files/"
MESH_INFO = {
    "axis_order": "xyz",
    "units": "millimeter",
    "voxel_size_xyz": [0.02, 0.02, 0.04],
    "downsample": 4,
    "classes": [
        {"value": 2, "name": "bone", "triangles": 12, "files": {"obj": "2.obj"}},
        {"value": 3, "name": "matrix", "triangles": 24, "files": {"obj": "3.obj"}},
    ],
}
# 80 x 60 x 40 voxels (x, y, z) of 0.02 x 0.02 x 0.04 mm.
MANIFEST = {
    "shape_czyx": [1, 40, 60, 80],
    "voxel_size_zyx": [0.04, 0.02, 0.02],
    "unit": "millimeter",
    "window": [10, 200],
}


def state_of(link: str) -> dict:
    """The state in a link's fragment, as Neuroglancer reads it."""
    assert link.startswith("/neuroglancer/?v=obj1#!")
    return json.loads(unquote(link.split("#!", 1)[1]))


def everything(manifest=MANIFEST, **changes) -> str:
    """A link with every layer there can be, for `manifest`, unless `changes` say otherwise."""
    arguments = {
        "labels_url": LABELS,
        "classes": CLASSES,
        "prediction_url": PREDICTION,
        "segmentation_url": SEGMENTATION,
    }
    return neuroglancer_link(PUBLIC, IMAGE, manifest, **{**arguments, **changes})


def layer_names(link: str) -> list[str]:
    return [layer["name"] for layer in state_of(link)["layers"]]


def source_url(layer: dict) -> str:
    source = layer["source"]
    return source if isinstance(source, str) else source["url"]


# --- the link --------------------------------------------------------------


def test_the_link_has_the_image_the_labels_and_the_results_as_layers():
    state = state_of(everything())
    assert state["layout"] == "4panel"
    image, labels, prediction, final = state["layers"]
    assert [layer["name"] for layer in state["layers"]] == [
        "image",
        "labels",
        "prediction",
        "final segmentation",
    ]
    assert image["type"] == "image"
    assert image["source"] == f"zarr3://{PUBLIC}{IMAGE}"
    assert image["shaderControls"] == {"normalized": {"range": [10, 200]}}
    # The labels are the group, so that levels of it are one volume.
    assert labels["type"] == "segmentation"
    assert labels["source"] == f"zarr3://{PUBLIC}{LABELS}"
    # The results are the `class` array of each group.
    assert prediction["type"] == final["type"] == "segmentation"
    assert source_url(prediction) == f"zarr3://{PUBLIC}{PREDICTION}class"
    assert source_url(final) == f"zarr3://{PUBLIC}{SEGMENTATION}class"


def test_the_labels_show_and_the_results_wait_to_be_switched_on():
    image, labels, prediction, final = state_of(everything())["layers"]
    for layer in (image, labels):
        assert layer.get("visible", True) is True
    assert prediction["visible"] is False
    assert final["visible"] is False
    # As in the annotator, labels are the most opaque, then the final segmentation.
    assert (labels["selectedAlpha"], prediction["selectedAlpha"]) == (0.5, 0.35)
    assert final["selectedAlpha"] == 0.45


def test_only_the_classes_are_selected_and_they_have_their_colors():
    classes = [
        (3, "#3D9BE8"),
        (2, "#e8a33d"),
        (1, "#000000"),
        (0, "#ffffff"),
        (255, "#123456"),
    ]
    labels = state_of(everything(classes=classes))["layers"][1]
    # Background (1) and unlabeled (0) are left out, so they stay transparent,
    # and nothing past the last class.
    assert labels["segments"] == [2, 3]
    assert labels["segmentColors"] == {"2": "#e8a33d", "3": "#3d9be8"}
    assert "ignoreNullVisibleSet" not in labels


def test_a_color_neuroglancer_would_refuse_is_left_to_its_default():
    labels = state_of(everything(classes=[(2, "red"), (3, "#3d9be8"), (4, "#12345")]))[
        "layers"
    ][1]
    assert labels["segments"] == [2, 3, 4]
    assert labels["segmentColors"] == {"3": "#3d9be8"}


def test_the_results_share_the_labels_selection_and_colors():
    _, labels, prediction, final = state_of(everything())["layers"]
    assert "segments" in labels and "segmentColors" in labels
    for layer in (prediction, final):
        assert layer["linkedSegmentationGroup"] == "labels"
        assert "segments" not in layer and "segmentColors" not in layer


def test_without_a_prediction_or_a_final_segmentation_there_are_no_such_layers():
    assert layer_names(everything(prediction_url=None)) == [
        "image",
        "labels",
        "final segmentation",
    ]
    assert layer_names(everything(segmentation_url=None)) == [
        "image",
        "labels",
        "prediction",
    ]
    assert layer_names(everything(prediction_url=None, segmentation_url=None)) == [
        "image",
        "labels",
    ]
    assert layer_names(neuroglancer_link(PUBLIC, IMAGE, MANIFEST)) == ["image"]


def test_the_first_segmentation_layer_holds_the_selection_whatever_it_is():
    layers = state_of(everything(labels_url=None))["layers"]
    prediction, final = layers[1:]
    assert prediction["name"] == "prediction"
    assert prediction["segments"] == [2, 3]
    assert prediction["segmentColors"] == {"2": "#e8a33d", "3": "#3d9be8"}
    assert final["linkedSegmentationGroup"] == "prediction"


def test_with_no_classes_nothing_is_drawn_not_everything():
    # Neuroglancer draws every segment when none is selected, background too,
    # unless a layer is told not to.
    layers = state_of(everything(classes=[]))["layers"][1:]
    assert len(layers) == 3
    for layer in layers:
        assert layer["ignoreNullVisibleSet"] is False
    assert layers[0]["segments"] == []
    assert "segmentColors" not in layers[0]


def test_the_image_alone_is_a_link_to_the_image():
    # How the image's own link (`GET /image`) is made.
    state = state_of(neuroglancer_link(PUBLIC, IMAGE, {"shape_czyx": [1, 4, 6, 8]}))
    [layer] = state["layers"]
    assert layer == {
        "type": "image",
        "name": "image",
        "source": f"zarr3://{PUBLIC}{IMAGE}",
    }
    state = state_of(neuroglancer_link(PUBLIC, IMAGE, MANIFEST))
    assert state["layers"][0]["shaderControls"] == {"normalized": {"range": [10, 200]}}


def test_existing_meshes_are_visible_colored_obj_layers_with_opacity_controls():
    state = state_of(everything(meshes_url=MESHES, meshes_manifest=MESH_INFO))
    meshes = [layer for layer in state["layers"] if layer["type"] == "mesh"]
    assert [layer["name"] for layer in meshes] == ["mesh: bone", "mesh: matrix"]
    assert [source_url(layer) for layer in meshes] == [
        f"obj://{PUBLIC}{MESHES}2.obj",
        f"obj://{PUBLIC}{MESHES}3.obj",
    ]
    for layer, (_, color) in zip(meshes, CLASSES, strict=True):
        assert layer.get("visible", True)
        assert color.lower() in layer["shader"]
        assert "opacity slider" in layer["shader"]
        assert "emitRGBA" in layer["shader"]


@pytest.mark.parametrize("unit", ["millimeter", "micrometer", None, "unknown"])
def test_mesh_coordinates_align_in_anisotropic_physical_or_unitless_images(unit):
    manifest = {**MANIFEST, "unit": unit}
    state = state_of(everything(manifest, meshes_url=MESHES, meshes_manifest=MESH_INFO))
    transform = state["layers"][-1]["source"]["transform"]
    assert (
        transform["inputDimensions"]
        == transform["outputDimensions"]
        == state["dimensions"]
    )
    # A vertex at (10,20,30) full-res voxels was exported as (.2,.4,1.2).
    # Neither the 4x meshing downsample nor physical spacing is applied twice.
    vertex = [0.2, 0.4, 1.2, 1]
    assert [
        sum(a * b for a, b in zip(row, vertex, strict=True))
        for row in transform["matrix"]
    ] == pytest.approx([10, 20, 30])


def test_meshes_ignore_deleted_empty_or_missing_obj_classes():
    info = {
        **MESH_INFO,
        "classes": [
            MESH_INFO["classes"][0],
            {"value": 3, "triangles": 0, "files": {"obj": "3.obj"}},
            {"value": 4, "triangles": 12, "files": {"obj": "4.obj"}},
            {"value": 3, "triangles": 12, "files": {"glb": "3.glb"}},
        ],
    }
    state = state_of(everything(meshes_url=MESHES, meshes_manifest=info))
    assert [layer["name"] for layer in state["layers"] if layer["type"] == "mesh"] == [
        "mesh: bone"
    ]


@pytest.mark.parametrize(
    "bad",
    [
        {"axis_order": "zyx"},
        {"voxel_size_xyz": [0, 1, 1]},
        {"voxel_size_xyz": [1, 2]},
        {"voxel_size_xyz": None},
        {"voxel_size_xyz": [float("inf"), 1, 1]},
    ],
)
def test_meshes_with_unknown_coordinates_are_not_overlaid(bad):
    state = state_of(
        everything(meshes_url=MESHES, meshes_manifest={**MESH_INFO, **bad})
    )
    assert all(layer["type"] != "mesh" for layer in state["layers"])


@pytest.mark.parametrize(
    "filename", ["../2.obj", "https://elsewhere/2.obj", "2.obj#fragment", "2.obj?query"]
)
def test_mesh_manifest_cannot_redirect_the_browser(filename):
    entry = {**MESH_INFO["classes"][0], "files": {"obj": filename}}
    state = state_of(
        everything(meshes_url=MESHES, meshes_manifest={**MESH_INFO, "classes": [entry]})
    )
    assert all(layer["type"] != "mesh" for layer in state["layers"])


@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {"window": [10, 100]},
        {"shape_czyx": None},
        {"shape_czyx": [1, 4, 6]},
        {"shape_czyx": [1, 0, 6, 8]},
    ],
)
def test_a_manifest_without_a_shape_gets_a_link_that_leaves_the_view_to_neuroglancer(
    manifest,
):
    state = state_of(neuroglancer_link(PUBLIC, IMAGE, manifest))
    assert "position" not in state and "crossSectionScale" not in state
    assert [layer["name"] for layer in state["layers"]] == ["image"]


def test_the_state_is_json_in_the_fragment_and_a_url_is_safe_with_it():
    link = everything()
    fragment = link.split("#!", 1)[1]
    # Nothing in it ends a fragment or a link early, and it round-trips.
    assert re.fullmatch(r"[A-Za-z0-9%:,/._~-]+", fragment)
    assert json.loads(unquote(fragment)) == state_of(link)
    assert link.count("#") == 1


def test_the_view_starts_in_the_middle_of_the_image_with_all_of_it_in_view():
    state = state_of(everything())
    assert state["position"] == [40, 30, 20]
    # Dimensions are named, in x, y, z (so the top left panel is the slice
    # across z, as in the annotator), in meters when the voxel size is known.
    assert list(state["dimensions"]) == ["x", "y", "z"]
    assert state["dimensions"] == {
        "x": [pytest.approx(2e-5), "m"],
        "y": [pytest.approx(2e-5), "m"],
        "z": [pytest.approx(4e-5), "m"],
    }
    # The longest side (80 voxels of the finest size, 2e-5 m: the 40 deep ones are 80 too) at 400 pixels.
    assert state["crossSectionScale"] == 0.2


def test_the_middle_of_an_odd_side_is_between_voxels_and_a_whole_voxel_is_not():
    manifest = {"shape_czyx": [1, 7, 10, 5]}
    assert state_of(neuroglancer_link(PUBLIC, IMAGE, manifest))["position"] == [
        2.5,
        5,
        3.5,
    ]


def test_arrays_without_a_voxel_size_are_told_the_images():
    _, labels, prediction, final = state_of(everything())["layers"]
    dimensions = state_of(everything())["dimensions"]
    # The labels group says it in its own metadata; the arrays of results don't.
    assert isinstance(labels["source"], str)
    for layer in (prediction, final):
        assert layer["source"]["transform"] == {
            "inputDimensions": dimensions,
            "outputDimensions": dimensions,
        }


@pytest.mark.parametrize(
    ("unit", "meters"),
    [
        ("meter", 1.0),
        ("millimeter", 1e-3),
        ("micrometer", 1e-6),
        ("nanometer", 1e-9),
        ("centimeter", 1e-2),
        ("kilometer", 1e3),
        ("angstrom", 1e-10),
        ("inch", 0.0254),
    ],
)
def test_the_voxel_size_is_given_in_meters(unit, meters):
    manifest = {**MANIFEST, "unit": unit, "voxel_size_zyx": [3, 2, 1]}
    dimensions = state_of(neuroglancer_link(PUBLIC, IMAGE, manifest))["dimensions"]
    assert dimensions == {
        "x": [pytest.approx(1 * meters), "m"],
        "y": [pytest.approx(2 * meters), "m"],
        "z": [pytest.approx(3 * meters), "m"],
    }


@pytest.mark.parametrize(
    "voxel",
    [
        {},
        {"unit": None, "voxel_size_zyx": None},
        {"unit": "millimeter", "voxel_size_zyx": None},
        {"unit": None, "voxel_size_zyx": [1, 1, 1]},
        # Not lengths, or not units Neuroglancer reads, or nonsense.
        {"unit": "second", "voxel_size_zyx": [1, 1, 1]},
        {"unit": "furlong", "voxel_size_zyx": [1, 1, 1]},
        {"unit": "milli", "voxel_size_zyx": [1, 1, 1]},
        {"unit": 7, "voxel_size_zyx": [1, 1, 1]},
        {"unit": "millimeter", "voxel_size_zyx": [1, 0, 1]},
        {"unit": "millimeter", "voxel_size_zyx": [1, -2, 1]},
        {"unit": "millimeter", "voxel_size_zyx": [1, 1]},
        {"unit": "millimeter", "voxel_size_zyx": 5},
        {"unit": "millimeter", "voxel_size_zyx": "big"},
        {"unit": "millimeter", "voxel_size_zyx": [1, 1, float("inf")]},
        {"unit": "millimeter", "voxel_size_zyx": [1, 1, float("nan")]},
    ],
)
def test_without_a_voxel_size_the_voxel_is_the_unit(voxel):
    manifest = {"shape_czyx": [1, 40, 60, 80], **voxel}
    state = state_of(everything(manifest))
    assert state["dimensions"] == {axis: [1, ""] for axis in "xyz"}
    # Nothing to tell the arrays of results: they are in voxels like the image.
    for layer in state["layers"][2:]:
        assert isinstance(layer["source"], str)
    assert state["position"] == [40, 30, 20]
    assert state["crossSectionScale"] == 0.2


def test_the_zoom_fits_the_longest_side_in_physical_size():
    def scale(shape, voxel):
        manifest = {
            "shape_czyx": [1, *shape],
            "voxel_size_zyx": voxel,
            "unit": "millimeter",
        }
        return state_of(neuroglancer_link(PUBLIC, IMAGE, manifest))["crossSectionScale"]

    # (z, y, x) voxels and their size: 400 pixels for the longest side, in the finest voxel's size.
    assert scale((100, 100, 100), [0.01] * 3) == 0.25
    assert scale((400, 100, 100), [0.01] * 3) == 1
    assert scale((1000, 3000, 2000), [0.01] * 3) == 7.5
    # Slices 4 times as far apart: z is the longest side, however few they are.
    assert scale((100, 100, 100), [0.04, 0.01, 0.01]) == 1
    # A single slice.
    assert scale((1, 512, 640), [0.02] * 3) == 1.6


def test_the_link_is_short_enough_to_share_with_every_class_there_can_be():
    classes = [
        (
            value,
            f"#{(value * 37) % 256:02x}{(value * 91) % 256:02x}{(value * 53) % 256:02x}",
        )
        for value in range(2, 255)
    ]
    link = everything(classes=classes)
    state = state_of(link)
    labels = state["layers"][1]
    assert labels["segments"] == list(range(2, 255))
    assert len(labels["segmentColors"]) == 253
    # The selection and colors are in the link once, not once for each layer.
    assert link.count("segmentColors") == 1
    assert unquote(link).count('"segments"') == 1
    # Percent-encoded, a project with all 253 classes still comes to under
    # 10 KB (a few classes, under 2 KB).
    assert len(link) < 10_000
    assert len(everything()) < 2_000


# --- the endpoint ----------------------------------------------------------


@pytest.fixture
def neuroglancer(tmp_path):
    directory = tmp_path / "neuroglancer"
    directory.mkdir()
    (directory / "index.html").write_text("<html>neuroglancer</html>")
    (directory / "main.bundle.js").write_text("console.log('ng')")
    return directory


@pytest.fixture
def with_neuroglancer(settings, neuroglancer):
    return settings.model_copy(update={"neuroglancer_dir": neuroglancer})


def make_project(browser, name="Skull") -> str:
    return browser.post("/api/projects", json={"name": name}).json()["id"]


def add_artifact(database_url, project, kind, manifest, inputs=None) -> str:
    """A committed artifact that is its slot's head, as a pipeline leaves one."""

    async def create(db):
        artifact = await artifacts.create_staging(
            db,
            project_id=uuid.UUID(project),
            kind=kind,
            head_slot=kind,
            inputs=inputs or {},
        )
        artifact.state = "committed"
        artifact.manifest = manifest
        await artifacts.set_head(db, artifact)
        return str(artifact.id)

    return run_db(database_url, create)


def add_class(browser, project, name, color) -> int:
    response = browser.post(
        f"/api/projects/{project}/labels/classes", json={"name": name, "color": color}
    )
    assert response.status_code == 201, response.text
    return response.json()["value"]


def link_of(browser, project) -> dict:
    response = browser.get(f"/api/projects/{project}/neuroglancer")
    assert response.status_code == 200, response.text
    assert list(response.json()) == ["url"]
    return state_of(response.json()["url"])


def test_members_get_a_link_to_all_of_their_projects_data(
    new_browser, with_neuroglancer, migrated_database_url
):
    ada = new_browser(with_neuroglancer)
    signup(ada)
    project = make_project(ada)
    image = add_artifact(migrated_database_url, project, "image", MANIFEST)
    bone = add_class(ada, project, "bone", "#E8A33D")
    matrix = add_class(ada, project, "matrix", "#3d9be8")
    prediction = add_artifact(
        migrated_database_url,
        project,
        "prediction",
        {"shape_zyx": [40, 60, 80]},
        inputs={"image_artifact_id": image},
    )
    segmentation = add_artifact(
        migrated_database_url,
        project,
        "segmentation",
        {"shape_zyx": [40, 60, 80]},
        inputs={"prediction_artifact_id": prediction},
    )

    state = link_of(ada, project)
    public = with_neuroglancer.public_url
    gateway = f"/api/projects/{project}/artifacts"
    layers = {layer["name"]: layer for layer in state["layers"]}
    assert list(layers) == ["image", "labels", "prediction", "final segmentation"]
    assert layers["image"]["source"] == f"zarr3://{public}{gateway}/{image}/zarr/"
    assert layers["labels"]["source"] == (
        f"zarr3://{public}/api/projects/{project}/labels/zarr/"
    )
    assert source_url(layers["prediction"]) == (
        f"zarr3://{public}{gateway}/{prediction}/zarr/class"
    )
    assert source_url(layers["final segmentation"]) == (
        f"zarr3://{public}{gateway}/{segmentation}/zarr/class"
    )
    # The project's classes, with the colors the API gives them.
    assert layers["labels"]["segments"] == [bone, matrix]
    assert layers["labels"]["segmentColors"] == {
        str(bone): "#e8a33d",
        str(matrix): "#3d9be8",
    }
    # The link is to this server's Neuroglancer, which is what the tab frames.
    assert (
        ada.get(f"/api/projects/{project}/neuroglancer")
        .json()["url"]
        .startswith("/neuroglancer/?v=obj1#!")
    )


def test_a_project_with_only_an_image_gets_the_image_and_its_labels(
    new_browser, with_neuroglancer, migrated_database_url
):
    ada = new_browser(with_neuroglancer)
    signup(ada)
    project = make_project(ada)
    add_artifact(migrated_database_url, project, "image", MANIFEST)
    state = link_of(ada, project)
    assert [layer["name"] for layer in state["layers"]] == ["image", "labels"]
    # No classes yet: the labels layer draws nothing, not everything.
    labels = state["layers"][1]
    assert labels["segments"] == []
    assert labels["ignoreNullVisibleSet"] is False


@pytest.mark.parametrize("direct_reference", [False, True])
def test_mesh_exports_load_without_regeneration_and_not_on_a_replacement_scan(
    new_browser, migrated_database_url, direct_reference
):
    ada = new_browser()
    signup(ada)
    project = make_project(ada)
    image = add_artifact(migrated_database_url, project, "image", MANIFEST)
    add_class(ada, project, "bone", "#e8a33d")
    prediction = add_artifact(
        migrated_database_url,
        project,
        "prediction",
        {"shape_zyx": [40, 60, 80]},
        {"image_artifact_id": image},
    )
    segmentation = add_artifact(
        migrated_database_url,
        project,
        "segmentation",
        {"shape_zyx": [40, 60, 80]},
        {"prediction_artifact_id": prediction},
    )
    inputs = {"segmentation_artifact_id": segmentation}
    if direct_reference:
        inputs["image_artifact_id"] = image
    meshes = add_artifact(migrated_database_url, project, "meshes", MESH_INFO, inputs)
    # Still load the saved mesh when a newer segmentation is made for this scan.
    add_artifact(
        migrated_database_url,
        project,
        "segmentation",
        {"shape_zyx": [40, 60, 80]},
        {"prediction_artifact_id": prediction},
    )
    pipelines_before = ada.get(f"/api/projects/{project}/pipelines").json()
    layer = next(
        layer for layer in link_of(ada, project)["layers"] if layer["type"] == "mesh"
    )
    assert f"/artifacts/{meshes}/files/2.obj" in source_url(layer)
    assert ada.get(f"/api/projects/{project}/pipelines").json() == pipelines_before
    # Shape and spacing are identical: only provenance reveals the mismatch.
    add_artifact(migrated_database_url, project, "image", MANIFEST)
    assert all(layer["type"] != "mesh" for layer in link_of(ada, project)["layers"])


def test_meshes_with_missing_or_foreign_provenance_are_left_out(
    new_browser, migrated_database_url
):
    ada = new_browser()
    signup(ada)
    project = make_project(ada)
    image = add_artifact(migrated_database_url, project, "image", MANIFEST)
    add_class(ada, project, "bone", "#e8a33d")
    other = make_project(ada, "Other scan")
    prediction = add_artifact(
        migrated_database_url, other, "prediction", {}, {"image_artifact_id": image}
    )
    segmentation = add_artifact(
        migrated_database_url,
        other,
        "segmentation",
        {},
        {"prediction_artifact_id": prediction},
    )
    for inputs in (
        {},
        {"segmentation_artifact_id": "invalid"},
        {"segmentation_artifact_id": str(uuid.uuid4())},
        {"segmentation_artifact_id": segmentation},
    ):
        add_artifact(migrated_database_url, project, "meshes", MESH_INFO, inputs)
        assert all(layer["type"] != "mesh" for layer in link_of(ada, project)["layers"])


def test_a_class_that_was_deleted_is_not_selected(
    new_browser, with_neuroglancer, migrated_database_url
):
    ada = new_browser(with_neuroglancer)
    signup(ada)
    project = make_project(ada)
    add_artifact(migrated_database_url, project, "image", MANIFEST)
    bone = add_class(ada, project, "bone", "#e8a33d")
    matrix = add_class(ada, project, "matrix", "#3d9be8")
    removed = ada.request("DELETE", f"/api/projects/{project}/labels/classes/{bone}")
    assert removed.status_code == 204
    labels = link_of(ada, project)["layers"][1]
    assert labels["segments"] == [matrix]
    assert labels["segmentColors"] == {str(matrix): "#3d9be8"}


def test_results_that_dont_fit_the_image_are_left_out(
    new_browser, with_neuroglancer, migrated_database_url
):
    ada = new_browser(with_neuroglancer)
    signup(ada)
    project = make_project(ada)
    add_artifact(migrated_database_url, project, "image", MANIFEST)
    # A prediction of another image (the image has been replaced since), and a
    # final segmentation of another shape.
    add_artifact(
        migrated_database_url,
        project,
        "prediction",
        {"shape_zyx": [40, 60, 80]},
        inputs={"image_artifact_id": str(uuid.uuid4())},
    )
    add_artifact(
        migrated_database_url, project, "segmentation", {"shape_zyx": [40, 60, 99]}
    )
    names = [layer["name"] for layer in link_of(ada, project)["layers"]]
    assert names == ["image", "labels"]
    # Predictions and segmentations that aren't finished have no manifest.
    add_artifact(migrated_database_url, project, "prediction", {}, inputs={})
    add_artifact(migrated_database_url, project, "segmentation", {})
    names = [layer["name"] for layer in link_of(ada, project)["layers"]]
    assert names == ["image", "labels"]


def test_only_members_get_a_link(new_browser, with_neuroglancer, migrated_database_url):
    ada = new_browser(with_neuroglancer)
    signup(ada)
    project = make_project(ada)
    add_artifact(migrated_database_url, project, "image", MANIFEST)
    bob = new_browser(with_neuroglancer)
    signup(bob, username="bob")
    url = f"/api/projects/{project}/neuroglancer"
    assert ada.get(url).status_code == 200
    # Other people's projects look like projects that aren't there.
    assert bob.get(url).status_code == 404
    assert bob.get(f"/api/projects/{uuid.uuid4()}/neuroglancer").status_code == 404
    assert new_browser(with_neuroglancer).get(url).status_code == 401
    # A member who is added can see it.
    added = ada.post(f"/api/projects/{project}/members", json={"username": "bob"})
    assert added.status_code == 201
    assert bob.get(url).status_code == 200


def test_a_project_without_an_image_has_nothing_to_show(new_browser, with_neuroglancer):
    ada = new_browser(with_neuroglancer)
    signup(ada)
    project = make_project(ada)
    response = ada.get(f"/api/projects/{project}/neuroglancer")
    assert response.status_code == 404
    assert "no image" in response.json()["detail"]


def test_every_server_has_a_neuroglancer_link(new_browser, migrated_database_url):
    ada = new_browser()
    signup(ada)
    project = make_project(ada)
    assert ada.get(f"/api/projects/{project}/neuroglancer").status_code == 404
    add_artifact(migrated_database_url, project, "image", MANIFEST)
    assert (
        ada.get(f"/api/projects/{project}/neuroglancer")
        .json()["url"]
        .startswith("/neuroglancer/?v=obj1#!")
    )
    assert ada.get("/neuroglancer/").status_code == 200
    # Other people's projects are still not theirs to ask about.
    bob = new_browser()
    signup(bob, username="bob")
    assert bob.get(f"/api/projects/{project}/neuroglancer").status_code == 404


def test_server_refuses_a_missing_or_unconfigured_neuroglancer(settings, tmp_path):
    empty = tmp_path / "nothing-built"
    empty.mkdir()
    for directory in (None, empty, tmp_path / "missing"):
        with pytest.raises(RuntimeError, match="Neuroglancer is required"):
            create_app(settings.model_copy(update={"neuroglancer_dir": directory}))


def test_serve_checks_neuroglancer_before_starting_workers(settings, monkeypatch):
    from ml4paleo_server.cli import main

    monkeypatch.setattr(
        "ml4paleo_server.settings.Settings",
        lambda: settings.model_copy(update={"neuroglancer_dir": None}),
    )
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: pytest.fail("Started workers"))
    with pytest.raises(RuntimeError, match="Neuroglancer is required"):
        main(["serve"])


# --- framing ---------------------------------------------------------------


def directives(policy: str) -> dict[str, list[str]]:
    parsed = {}
    for part in policy.split(";"):
        name, *sources = part.split()
        parsed[name] = sources
    return parsed


def frame_sources(policy: str) -> list[str]:
    """What a page with this policy may frame: `frame-src`, else `child-src`, else `default-src`."""
    found = directives(policy)
    for name in ("frame-src", "child-src", "default-src"):
        if name in found:
            return found[name]
    return ["*"]


def test_neuroglancer_may_be_framed_by_this_site_and_nothing_else_may_be_framed(
    new_browser, with_neuroglancer, migrated_database_url
):
    browser = new_browser(with_neuroglancer)
    signup(browser)
    project = make_project(browser)
    add_artifact(migrated_database_url, project, "image", MANIFEST)
    for path in [
        "/neuroglancer/",
        "/neuroglancer/main.bundle.js",
        "/neuroglancer/nope.js",
    ]:
        response = browser.get(path)
        policy = directives(response.headers["content-security-policy"])
        assert policy["frame-ancestors"] == ["'self'"], path
        # It still runs only what is its own, and talks only to this site, so
        # a link can't have it load data from anywhere else.
        assert policy["connect-src"] == ["'self'"], path
        assert policy["default-src"] == ["'self'"], path
        assert "'unsafe-inline'" not in policy["script-src"], path
    # Everything else, the web app and the API, may be framed by no one.
    for path in [
        "/",
        "/projects",
        f"/p/{project}/neuroglancer",
        f"/p/{project}/annotate",
        "/api/health",
        f"/api/projects/{project}",
        f"/api/projects/{project}/neuroglancer",
        f"/api/projects/{project}/labels/zarr/zarr.json",
        "/api/nope",
        # The folder without its slash is a redirect, which isn't Neuroglancer.
        "/neuroglancer",
    ]:
        response = browser.get(path, follow_redirects=False)
        policy = directives(response.headers["content-security-policy"])
        assert policy["frame-ancestors"] == ["'none'"], path
    # No older header says anything else.
    for path in ["/neuroglancer/", "/projects", "/api/health"]:
        assert "x-frame-options" not in browser.get(path).headers, path


@pytest.mark.parametrize(
    "path", ["/neuroglancer/", "/neuroglancer/index.html", "/neuroglancer/?v=obj1"]
)
def test_neuroglancer_entry_point_revalidates_after_an_upgrade(
    new_browser, with_neuroglancer, path
):
    browser = new_browser(with_neuroglancer)
    response = browser.get(path)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    cached = browser.get(path, headers={"If-None-Match": response.headers["etag"]})
    assert cached.status_code == 304
    assert cached.headers["cache-control"] == "no-cache"


def test_the_web_app_may_frame_the_neuroglancer_it_serves(
    new_browser, with_neuroglancer
):
    browser = new_browser(with_neuroglancer)
    app = browser.get("/projects").headers["content-security-policy"]
    # `frame-src` isn't set, so frames follow `default-src`: this origin.
    assert "frame-src" not in directives(app)
    assert frame_sources(app) == ["'self'"]
    # Every page of the web app has that policy, the tab among them.
    assert browser.get("/p/1/neuroglancer").headers["content-security-policy"] == app


def test_the_bundled_viewer_never_falls_through_to_the_spa(new_browser):
    browser = new_browser()
    assert browser.get("/neuroglancer/").text == "<html>neuroglancer</html>"
    assert browser.get("/neuroglancer/missing.js").status_code == 404
