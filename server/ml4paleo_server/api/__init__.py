"""
API routes, grouped by area. Every router's paths start with `/api`.
"""

from . import admin, admin_jobs, auth, me, projects, storage_proxy, worker

ROUTERS = [
    auth.router,
    admin.router,
    admin_jobs.router,
    me.router,
    projects.router,
    worker.router,
    storage_proxy.router,
]
