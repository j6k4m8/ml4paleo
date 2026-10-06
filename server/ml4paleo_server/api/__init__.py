"""
API routes, grouped by area. Every router's paths start with `/api`.
"""

from . import (
    admin,
    admin_jobs,
    auth,
    gateway,
    labels,
    me,
    models,
    pipelines,
    projects,
    rois,
    segmentation,
    storage_proxy,
    uploads,
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
    labels.router,
    rois.router,
    models.router,
    segmentation.router,
    worker.router,
    storage_proxy.router,
]
