"""
API routes, grouped by area. Every router's paths start with `/api`.
"""

from . import (
    admin,
    admin_jobs,
    auth,
    exports,
    gateway,
    labelimports,
    labels,
    me,
    meshes,
    models,
    pipelines,
    projects,
    rois,
    segmentation,
    storage_proxy,
    uploads,
    v1jobs,
    worker,
)

ROUTERS = [
    auth.router,
    admin.router,
    admin_jobs.router,
    me.router,
    projects.router,
    uploads.router,
    pipelines.router,
    gateway.router,
    gateway.files_router,
    labels.router,
    labelimports.router,
    rois.router,
    models.router,
    segmentation.router,
    meshes.router,
    exports.router,
    v1jobs.router,
    worker.router,
    storage_proxy.router,
]
