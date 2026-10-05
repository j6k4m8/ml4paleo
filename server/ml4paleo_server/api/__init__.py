"""
API routes, grouped by area. Every router's paths start with `/api`.
"""

from . import admin, auth, me, projects

ROUTERS = [auth.router, admin.router, me.router, projects.router]
