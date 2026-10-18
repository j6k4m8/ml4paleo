"""Optional feature readers for plugins that describe a feature recipe."""

from typing import cast

from ml4paleo.segmentation.dataset import ImageArray, tile_for
from ml4paleo.segmentation.features import FeatureRecipe, FeatureVolume, extractor
from ml4paleo.segmentation.plugin import CropCost

from .context import JobContext


def reader(
    ctx: JobContext,
    image: ImageArray,
    image_id: str | None,
    window: tuple[float, float],
    recipe: FeatureRecipe | None,
):
    if recipe is None or ctx.feature_cache is None or image_id is None:
        return None
    implementation = extractor(recipe)
    # Keep chunk generation within half of the job's budget, alongside
    # the caller's crop/output and model. Small workers use smaller chunks.
    cost = CropCost(
        implementation.halo,
        implementation.bytes_per_voxel(int(image.shape[0]), ctx.threads)
        + 16 * image.shape[0],
    )
    tile = max(16, tile_for(ctx.memory_budget_bytes // 2, cost, 16, 64) // 16 * 16)
    return FeatureVolume(
        image,
        image_id,
        window,
        recipe,
        ctx.feature_cache,
        ctx.threads,
        tile=tile,
        check=ctx.check,
    ).read


def recipe_for(plugin, params) -> FeatureRecipe | None:
    describe = getattr(plugin, "feature_recipe", None)
    return cast(FeatureRecipe, describe(params)) if callable(describe) else None
