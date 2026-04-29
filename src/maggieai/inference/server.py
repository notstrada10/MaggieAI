"""Launcher for the local MLX inference server (NATIVE macOS, NOT containerized).

Minimal wrapper around `mlx_lm.server`, which exposes an
OpenAI-compatible endpoint on `localhost:8001`.

Start:
    uv pip install -e ".[inference-local]"
    maggie-inference

Or directly:
    python -m mlx_lm.server --model mlx-community/Qwen2.5-14B-Instruct-4bit --port 8001

Important: this module requires `mlx-lm`, installed only by the
`inference-local` extra. If the import fails it means you are trying
to run local inference inside Docker/Linux — that is not supported
(Metal).
"""

from __future__ import annotations

import logging
import sys

from maggieai.config import get_settings

logger = logging.getLogger(__name__)


def run() -> None:
    settings = get_settings()
    try:
        from mlx_lm.server import main as mlx_server_main
    except ImportError as exc:  # pragma: no cover
        sys.stderr.write(
            "mlx-lm is not installed. Install with: uv pip install -e '.[inference-local]'\n"
            "Note: only runs on macOS Apple Silicon (Metal).\n"
        )
        raise SystemExit(1) from exc

    # mlx_lm.server reads arguments from sys.argv — build them ourselves.
    sys.argv = [
        "mlx_lm.server",
        "--model",
        settings.inference_local_model,
        "--port",
        str(settings.inference_local_port),
        "--host",
        "0.0.0.0",
    ]
    logger.info(
        "Starting mlx-lm server: %s on port %d",
        settings.inference_local_model,
        settings.inference_local_port,
    )
    mlx_server_main()


if __name__ == "__main__":
    run()
