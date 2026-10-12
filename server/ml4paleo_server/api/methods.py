"""Publication text from a result's pinned lineage, never the newest model."""

import datetime
import math
import uuid
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select

from .. import artifacts
from ..auth.deps import DbSession
from ..db import Artifact, Job, LabelClass, TrainedModel
from .projects import MemberProject

router = APIRouter(prefix="/api/projects/{project_id}/methods", tags=["results"])
DOI = "https://doi.org/10.3897/fr.28.e139379"
CREDIT = (
    "We used the ml4paleo software suite (During, Matelsky et al., "
    f"Fossil Record, 2025; {DOI}) to process volumetric image data."
)
Source = Literal["prediction", "segmentation", "meshes"]
SectionKey = Literal["training", "performance", "prediction", "segmentation", "meshes"]


class MethodsSection(BaseModel):
    key: SectionKey
    text: str
    warnings: list[str] = Field(default_factory=list)


class MethodsOut(BaseModel):
    source: Source
    artifact_id: uuid.UUID
    committed_at: datetime.datetime
    model_id: uuid.UUID | None
    paragraph: str
    warnings: list[str]
    credit: str
    sections: list[MethodsSection]
    lineage_warnings: list[str] = Field(default_factory=list)


async def _artifact(db, project_id, value, kind) -> Artifact | None:
    try:
        wanted = uuid.UUID(str(value))
    except ValueError:
        return None
    # Even corrupt legacy lineage must never disclose another project's data.
    return await db.scalar(
        select(Artifact).where(
            Artifact.id == wanted,
            Artifact.project_id == project_id,
            Artifact.kind == kind,
        )
    )


def _metrics(model: TrainedModel, manifest: dict) -> dict:
    # Both records belong to this exact model. Older manifests may omit fields
    # that are present in its database record; explicit manifest values win.
    return {**(model.metrics or {}), **(manifest.get("metrics") or {})}


def model_sentences(
    model: TrainedModel, manifest: dict, *, include_performance: bool = True
) -> tuple[list[str], list[str]]:
    params = manifest.get("params", model.params)
    plugin = manifest.get("plugin", model.plugin)
    version = manifest.get("plugin_version", model.plugin_version)
    metrics = _metrics(model, manifest)
    warnings = []
    if plugin == "rf":
        settings = []
        if "n_estimators" in params:
            settings.append(f"{params['n_estimators']} trees")
        if "max_depth" in params:
            settings.append(f"a maximum tree depth of {params['max_depth']}")
        if "seed" in params:
            settings.append(f"random seed {params['seed']}")
        sentences = [
            "We trained a random forest classifier"
            + (" with " + ", ".join(settings) if settings else "")
            + "."
        ]
        # The v2 RF plugin computes these 3D features, unlike the v1 slice model.
        if (
            version in ("1", "1.0", "1.0.0", "2", "2.0", "2.0.0")
            and "sigma_max" in params
        ):
            sentences.append(
                "Input features comprised multiscale 3D intensity, edge and "
                "Hessian-based texture features for each image channel, with "
                f"Gaussian scales from 1 to {params['sigma_max']:g} voxels."
            )
        cap = metrics.get("samples_per_class_used", params.get("samples_per_class"))
        if cap is not None:
            sentences.append(
                "Training voxels were sampled from the saved labels using "
                f"per-class reservoir sampling, capped at {cap:,} voxels per class."
            )
    else:
        settings = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        sentences = [
            f"We trained the {plugin} model"
            + (f" (plugin version {version})" if version else "")
            + (f" with {settings}" if settings else "")
            + "."
        ]
    samples = manifest.get("samples")
    if samples:
        sentences.append(
            f"The training set comprised {sum(samples.values()):,} sampled labeled voxels."
        )
    if include_performance:
        performance, notices = performance_sentences(metrics)
        sentences.extend(performance)
        warnings.extend(notices)
    return sentences, warnings


def _score(value) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def performance_sentences(
    metrics: dict, class_names: dict[str, str] | None = None
) -> tuple[list[str], list[str]]:
    sentences: list[str] = []
    warnings: list[str] = []
    class_names = class_names or {}
    evaluated = metrics.get("evaluation_voxels", metrics.get("voxels", 0))
    evaluated = (
        evaluated
        if isinstance(evaluated, int)
        and not isinstance(evaluated, bool)
        and evaluated > 0
        else 0
    )
    scores = []
    for key, label in [
        ("mean_dice", "mean Dice"),
        ("mean_iou", "mean intersection-over-union (IoU)"),
    ]:
        if _score(metrics.get(key)):
            scores.append(f"{label} {metrics[key]:.3f}")
    if _score(metrics.get("accuracy")):
        scores.append(f"voxel accuracy {metrics['accuracy']:.2%}")
    per_class = []
    for value, score in sorted((metrics.get("classes") or {}).items()):
        if isinstance(score, dict) and _score(score.get("dice")):
            name = class_names.get(str(value), f"class {value}")
            per_class.append(f"{name}: {score['dice']:.3f}")
    has_scores = bool(scores or per_class)
    population = f"{evaluated:,} " if evaluated else ""
    if (evaluated or has_scores) and metrics.get(
        "evaluation"
    ) == "human_validation_rois":
        sentences.append(
            f"Performance was evaluated on {population}human-annotated voxels "
            "in held-out validation regions excluded from model fitting."
        )
    elif (evaluated or has_scores) and metrics.get("evaluation") == "human_out_of_bag":
        sentences.append(
            f"Performance was estimated from out-of-bag predictions on {population}"
            "human-annotated training voxels, using only trees whose bootstrap "
            "samples excluded each evaluated voxel; this was an internal estimate, "
            "not an independent spatial validation."
        )
    elif evaluated or has_scores:
        if evaluated:
            sentences.append(
                f"Recorded model performance was measured on {evaluated:,} labeled voxels."
            )
        # Legacy scores are still useful, but don't invent their split strategy.
        warnings.append(
            "The saved scores do not record the validation method. Confirm how they were measured before publication."
        )
    if scores:
        sentences.append("The recorded scores were " + ", ".join(scores) + ".")
    if per_class:
        sentences.append("Per-class Dice scores were " + "; ".join(per_class) + ".")
    if not has_scores:
        warnings.append(
            "No Dice, IoU or accuracy scores are recorded for this saved model."
        )
    return sentences, warnings


async def describe(db, project_id, source: Source, result: Artifact) -> MethodsOut:
    sections: list[MethodsSection] = []
    sentences: list[str] = []
    warnings = []
    segmentation = result if source == "segmentation" else None
    prediction = result if source == "prediction" else None
    if source == "meshes":
        segmentation = await _artifact(
            db,
            project_id,
            result.inputs.get("segmentation_artifact_id"),
            "segmentation",
        )
    if segmentation is not None:
        prediction = await _artifact(
            db,
            project_id,
            segmentation.inputs.get("prediction_artifact_id"),
            "prediction",
        )
    model = None
    model_source = prediction or segmentation
    if model_source:
        try:
            model_id = uuid.UUID(str(model_source.inputs.get("model_id")))
        except ValueError:
            pass
        else:
            # Soft-deleted models retain their provenance and are still usable here.
            model = await db.scalar(
                select(TrainedModel).where(
                    TrainedModel.id == model_id,
                    TrainedModel.project_id == project_id,
                )
            )
    if model:
        model_artifact = await _artifact(db, project_id, model.artifact_id, "model")
        manifest = (model_artifact.manifest if model_artifact else None) or {}
        text, notices = model_sentences(model, manifest, include_performance=False)
        sections.append(
            MethodsSection(key="training", text=" ".join(text), warnings=notices)
        )
        names = {
            str(value): name
            for value, name in (
                await db.execute(
                    select(LabelClass.value, LabelClass.name).where(
                        LabelClass.project_id == project_id
                    )
                )
            ).all()
        }
        names["1"] = "Background"
        text, notices = performance_sentences(_metrics(model, manifest), names)
        sections.append(
            MethodsSection(key="performance", text=" ".join(text), warnings=notices)
        )
        sections.append(
            MethodsSection(
                key="prediction",
                text="The trained model was applied to the image volume in chunks to generate voxel-wise class predictions.",
            )
        )
    else:
        warnings.append(
            "The source model's training details are unavailable (for example, for an imported result). Add them before publication."
        )
    if segmentation is not None and "min_voxels" in segmentation.inputs:
        sentences.append(
            "Final segmentation combined model predictions with saved annotations, which took precedence over predictions."
        )
        minimum = segmentation.inputs["min_voxels"]
        if minimum > 0:
            sentences.append(
                f"Connected components smaller than {minimum:,} voxels were removed unless they contained labeled voxels."
            )
        sections.append(MethodsSection(key="segmentation", text=" ".join(sentences)))
    sentences = []
    if source == "meshes":
        info = result.manifest or {}
        job = (
            await db.get(Job, result.produced_by_job)
            if result.produced_by_job
            else None
        )
        native = info.get("mesher") == "zmesh" or (
            job is not None and job.kind == "mesh.finalize"
        )
        sentences.append(
            "Surface meshes were generated"
            + (
                " with zmesh in overlapping chunks, with shared boundary vertices welded during assembly."
                if native
                else " from the segmentation."
            )
        )
        if not native:
            warnings.append(
                "The meshing implementation is not recorded for this result; confirm it before publication."
            )
        downsample = result.inputs.get("downsample", info.get("downsample"))
        method = result.inputs.get("method", info.get("method"))
        simplify = result.inputs.get("simplify", info.get("simplify"))
        if downsample and downsample > 1:
            rule = {"any": "any-voxel occupancy", "majority": "majority occupancy"}.get(
                str(method), str(method)
            )
            sentences.append(
                f"Meshing used a {downsample}-fold downsampling along each axis with {rule}."
            )
        elif downsample == 1:
            sentences.append("Meshing used the full voxel resolution.")
        if simplify is not None:
            sentences.append(
                f"Mesh simplification used a tolerance of {simplify:g} mesh-resolution voxels."
                if simplify > 0
                else "No mesh simplification was applied."
            )
        if info.get("units") and info["units"] != "voxels":
            sentences.append(
                f"Mesh coordinates were scaled by the recorded voxel spacing and expressed in {info['units']} units."
            )
        sections.append(MethodsSection(key="meshes", text=" ".join(sentences)))
    return MethodsOut(
        source=source,
        artifact_id=result.id,
        committed_at=result.state_changed_at,
        model_id=model.id if model and model.deleted_at is None else None,
        paragraph=" ".join(
            [CREDIT, *(section.text for section in sections if section.text)]
        ),
        warnings=warnings
        + [warning for section in sections for warning in section.warnings],
        credit=CREDIT,
        sections=sections,
        lineage_warnings=warnings,
    )


@router.get("")
async def methods(project: MemberProject, db: DbSession) -> list[MethodsOut]:
    entries = []
    heads = {}
    for source in ("prediction", "segmentation", "meshes"):
        result = await artifacts.head(db, project.id, source)
        if result and result.manifest:
            heads[source] = result
            entries.append(await describe(db, project.id, source, result))
    for entry in entries:
        parent = {"segmentation": "prediction", "meshes": "segmentation"}.get(
            entry.source
        )
        if parent and parent in heads:
            linked = heads[entry.source].inputs.get(f"{parent}_artifact_id")
            if linked != str(heads[parent].id):
                subject = (
                    "These meshes use"
                    if entry.source == "meshes"
                    else "This segmentation uses"
                )
                entry.warnings.insert(
                    0,
                    f"{subject} an older {parent}. The paragraph describes that saved result, not the newer one.",
                )
                entry.lineage_warnings.insert(0, entry.warnings[0])
    return entries
