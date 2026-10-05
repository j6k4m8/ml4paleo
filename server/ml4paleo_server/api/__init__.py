"""
API routes, grouped by area. Every router's paths start with `/api`.
"""

from . import admin, auth

ROUTERS = [auth.router, admin.router]
