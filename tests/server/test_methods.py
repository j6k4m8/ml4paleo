"""Methods describe the selected result, not today's newest model or settings."""

import datetime
import uuid

import pytest
from helpers import run_db, signup
from ml4paleo_server import artifacts
from ml4paleo_server.api.methods import DOI, model_sentences, performance_sentences
from ml4paleo_server.db import Artifact, LabelClass, TrainedModel, TrainingSet


def model(**kwargs):
    return TrainedModel(
        plugin="rf",
        plugin_version="2",
        params={
            "n_estimators": 100,
            "max_depth": 16,
            "samples_per_class": 50000,
            "sigma_max": 8.0,
            "seed": 0,
        },
        **kwargs,
    )


@pytest.mark.parametrize(
    "evaluation, phrase",
    [
        ("human_out_of_bag", "not an independent spatial validation"),
        (
            "human_validation_rois",
            "held-out validation regions excluded from model fitting",
        ),
    ],
)
def test_validation_is_the_recorded_method(evaluation, phrase):
    sentences, warnings = model_sentences(
        model(
            metrics={
                "evaluation": evaluation,
                "evaluation_voxels": 321,
                "samples_per_class_used": 1000,
                "mean_dice": 0.875,
            }
        ),
        {},
    )
    paragraph = " ".join(sentences)
    assert phrase in paragraph
    assert "321 human-annotated" in paragraph
    assert "1,000 voxels per class" in paragraph
    assert "cross-validation" not in paragraph
    assert not warnings


def test_performance_includes_saved_dice_iou_accuracy_and_named_classes():
    sentences, warnings = performance_sentences(
        {
            "evaluation": "human_out_of_bag",
            "evaluation_voxels": 17586,
            "mean_dice": 0.946943,
            "mean_iou": 0.900415,
            "accuracy": 0.9542818,
            "classes": {"2": {"dice": 0.920671}, "3": {"dice": 0.973214}},
        },
        {"2": "Bone", "3": "Tooth"},
    )
    text = " ".join(sentences)
    for expected in [
        "17,586",
        "mean Dice 0.947",
        "(IoU) 0.900",
        "95.43%",
        "Bone: 0.921",
        "Tooth: 0.973",
        "out-of-bag",
    ]:
        assert expected in text
    assert not warnings


def test_legacy_metrics_are_reported_without_inventing_a_validation_method():
    sentences, warnings = performance_sentences(
        {"mean_dice": 0.72, "accuracy": 0.9, "voxels": 500}
    )
    text = " ".join(sentences)
    assert "mean Dice 0.720" in text and "90.00%" in text
    assert "500 labeled voxels" in text
    assert "held-out" not in text and "out-of-bag" not in text
    assert "validation method" in warnings[0]
    assert "No Dice" not in " ".join(warnings)


def test_missing_manifest_fields_use_the_same_models_database_metrics():
    sentences, _ = model_sentences(
        model(
            metrics={
                "mean_dice": 0.8,
                "mean_iou": 0.7,
                "evaluation": "human_out_of_bag",
                "evaluation_voxels": 10,
            }
        ),
        {"metrics": {"mean_dice": 0.9}},
    )
    text = " ".join(sentences)
    assert "mean Dice 0.900" in text and "(IoU) 0.700" in text
    assert "mean Dice 0.800" not in text


@pytest.mark.parametrize(
    "score", [None, True, "0.9", float("nan"), float("inf"), -0.1, 1.1]
)
def test_invalid_scores_are_not_published(score):
    sentences, warnings = performance_sentences({"mean_dice": score})
    assert not sentences
    assert "No Dice" in warnings[0]


def test_zero_is_a_real_score_not_a_missing_value():
    sentences, warnings = performance_sentences(
        {
            "evaluation": "human_validation_rois",
            "evaluation_voxels": 20,
            "mean_dice": 0,
            "accuracy": 0,
        }
    )
    text = " ".join(sentences)
    assert "mean Dice 0.000" in text and "0.00%" in text
    assert not warnings


def test_missing_validation_and_unknown_models_do_not_invent_methods():
    custom = TrainedModel(plugin="custom", plugin_version="7", params={"depth": 4})
    sentences, warnings = model_sentences(custom, {})
    assert "custom model (plugin version 7) with depth=4" in " ".join(sentences)
    assert "random forest" not in " ".join(sentences)
    assert warnings
    sentences, warnings = model_sentences(
        model(metrics={"evaluation": "human_out_of_bag", "evaluation_voxels": 0}), {}
    )
    assert "Performance" not in " ".join(sentences)
    assert warnings


def test_methods_follow_pinned_lineage_and_keep_deleted_model_details(
    new_browser, migrated_database_url
):
    browser = new_browser()
    signup(browser)
    pid = browser.post("/api/projects", json={"name": "Skull"}).json()["id"]
    assert browser.get(f"/api/projects/{pid}/methods").json() == []

    async def seed(db):
        project_id = uuid.UUID(pid)
        db.add(TrainingSet(id="a" * 64, project_id=project_id, summary={}))
        await db.flush()
        old_model = model(
            project_id=project_id,
            name="Original",
            training_set_id="a" * 64,
            class_values=[1, 2],
            metrics={
                "evaluation": "human_out_of_bag",
                "evaluation_voxels": 600,
                "mean_dice": 0.8123,
                "classes": {"2": {"dice": 0.8123}},
            },
            deleted_at=datetime.datetime.now(datetime.UTC),
        )
        new_model = model(
            project_id=project_id,
            name="Newer",
            training_set_id="a" * 64,
            class_values=[1, 2],
        )
        new_model.params = {"n_estimators": 499}
        new_model.metrics = {"mean_dice": 0.999}
        db.add(LabelClass(project_id=project_id, value=2, name="Bone", color="#ffffff"))
        db.add_all([old_model, new_model])
        await db.flush()

        async def result(kind, inputs, manifest=None):
            item = await artifacts.create_staging(
                db, project_id=project_id, kind=kind, head_slot=kind, inputs=inputs
            )
            item.state = "committed"
            item.manifest = manifest or {"kind": kind}
            await artifacts.set_head(db, item)
            return item

        prediction = await result("prediction", {"model_id": str(old_model.id)})
        segmentation = await result(
            "segmentation",
            {"prediction_artifact_id": str(prediction.id), "min_voxels": 50},
        )
        mesh = await result(
            "meshes",
            {
                "segmentation_artifact_id": str(segmentation.id),
                "downsample": 2,
                "method": "any",
                "simplify": 0.5,
            },
            {"kind": "meshes", "mesher": "zmesh", "units": "millimeter"},
        )
        newer = await result("prediction", {"model_id": str(new_model.id)})
        await result(
            "segmentation", {"prediction_artifact_id": str(newer.id), "min_voxels": 99}
        )
        return str(mesh.id)

    mesh_id = run_db(migrated_database_url, seed)
    response = browser.get(f"/api/projects/{pid}/methods")
    assert response.status_code == 200, response.text
    entries = {item["source"]: item for item in response.json()}
    old = entries["meshes"]
    text = old["paragraph"]
    assert old["artifact_id"] == mesh_id
    assert old["model_id"] is None  # don't offer a broken link to a deleted model
    for expected in [
        DOI,
        "2025",
        "100 trees",
        "600 human-annotated",
        "50 voxels",
        "zmesh",
        "2-fold",
        "0.5 mesh-resolution",
        "millimeter",
        "mean Dice 0.812",
        "Bone: 0.812",
    ]:
        assert expected in text
    assert "499 trees" not in text and "99 voxels" not in text
    assert "0.999" not in text
    sections = {section["key"]: section for section in old["sections"]}
    assert set(sections) == {
        "training",
        "performance",
        "prediction",
        "segmentation",
        "meshes",
    }
    assert "mean Dice 0.812" in sections["performance"]["text"]
    assert "mean Dice" not in sections["training"]["text"]
    assert old["paragraph"] == " ".join(
        [old["credit"], *(section["text"] for section in old["sections"])]
    )
    assert "older segmentation" in old["warnings"][0]
    assert "499 trees" in entries["segmentation"]["paragraph"]
    stranger = new_browser()
    signup(stranger, username="bob")
    assert stranger.get(f"/api/projects/{pid}/methods").status_code == 404


def test_imported_or_cross_project_lineage_is_not_fabricated(
    new_browser, migrated_database_url
):
    browser = new_browser()
    signup(browser)
    pid = browser.post("/api/projects", json={"name": "Imported"}).json()["id"]
    other = browser.post("/api/projects", json={"name": "Other"}).json()["id"]

    async def seed(db):
        foreign = Artifact(
            project_id=uuid.UUID(other),
            kind="segmentation",
            state="committed",
            inputs={"min_voxels": 12345},
            manifest={"kind": "segmentation"},
        )
        db.add(foreign)
        await db.flush()
        mesh = await artifacts.create_staging(
            db,
            project_id=uuid.UUID(pid),
            kind="meshes",
            head_slot="meshes",
            inputs={"segmentation_artifact_id": str(foreign.id)},
        )
        mesh.state = "committed"
        mesh.manifest = {"kind": "meshes", "units": "voxels"}
        await artifacts.set_head(db, mesh)

    run_db(migrated_database_url, seed)
    entry = browser.get(f"/api/projects/{pid}/methods").json()[0]
    assert "12,345" not in entry["paragraph"]
    assert "We trained" not in entry["paragraph"]
    assert "zmesh" not in entry["paragraph"]
    assert len(entry["warnings"]) == 2
