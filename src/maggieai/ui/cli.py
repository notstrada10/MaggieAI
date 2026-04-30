"""Console entry point for the Streamlit workbench.

Streamlit is not invoked the way other Python apps are: it does not run
``python -m`` on a script — it runs ``streamlit run <path>`` and bootstraps
its own runtime around the file. To expose this through a console script
we hand-roll ``sys.argv`` and call Streamlit's CLI main.

Usage:
    uv pip install -e ".[ui]"
    maggie-ui                       # opens http://localhost:8501
    maggie-ui --server.port 8502    # forward extra args to streamlit
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web.cli import main as streamlit_main

    app_path = str(Path(__file__).resolve().parent / "app.py")
    extra_args = sys.argv[1:]
    sys.argv = ["streamlit", "run", app_path, *extra_args]
    streamlit_main()


if __name__ == "__main__":
    main()
