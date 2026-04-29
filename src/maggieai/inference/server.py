"""Avvio del server inference MLX locale (NATIVO macOS, NON containerizzato).

Wrapper minimale attorno a `mlx_lm.server`, che espone un endpoint
OpenAI-compatibile su `localhost:8001`.

Avvio:
    uv pip install -e ".[inference-local]"
    maggie-inference

Oppure direttamente:
    python -m mlx_lm.server --model mlx-community/Qwen2.5-14B-Instruct-4bit --port 8001

Importante: questo modulo richiede `mlx-lm`, installato solo nel gruppo
`inference-local`. Se l'import fallisce significa che stai cercando di
girare l'inference locale dentro Docker/Linux — non è supportato (Metal).
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
            "mlx-lm non installato. Installa con: uv pip install -e '.[inference-local]'\n"
            "Nota: gira solo su macOS Apple Silicon (Metal).\n"
        )
        raise SystemExit(1) from exc

    # mlx_lm.server legge gli argomenti da sys.argv — costruiamoli noi.
    sys.argv = [
        "mlx_lm.server",
        "--model", settings.inference_local_model,
        "--port", str(settings.inference_local_port),
        "--host", "0.0.0.0",
    ]
    logger.info("Avvio mlx-lm server: %s su porta %d",
                settings.inference_local_model, settings.inference_local_port)
    mlx_server_main()


if __name__ == "__main__":
    run()
