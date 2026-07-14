from __future__ import annotations

import base64
import inspect

from pharma_financial.ui import shell


def test_pharma_hero_uses_bundled_png_data_uri() -> None:
    image_bytes = shell._HERO_IMAGE_PATH.read_bytes()

    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert shell._hero_image_data_uri().startswith("data:image/png;base64,")
    encoded = shell._hero_image_data_uri().split(",", 1)[1]
    assert base64.b64decode(encoded) == image_bytes


def test_pharma_hero_has_full_background_and_readable_overlay() -> None:
    source = inspect.getsource(shell.inject_app_theme) + inspect.getsource(
        shell.render_model_hero
    )

    assert 'class="designer-hero-image"' in source
    assert 'class="designer-hero-content"' in source
    assert "object-fit: cover" in source
    assert ".designer-hero::after" in source
    assert "--pharma-hero-overlay" in source
    assert "@media (max-width: 760px)" in source