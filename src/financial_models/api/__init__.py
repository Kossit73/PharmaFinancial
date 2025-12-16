"""FastAPI application exposing the Pharmaceuticals financial engine."""

from .server import app, create_app

__all__ = ["app", "create_app"]
