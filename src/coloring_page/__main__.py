"""Allows running the CLI via ``python -m coloring_page``."""

import sys

from coloring_page.cli import main

if __name__ == "__main__":
    sys.exit(main())
