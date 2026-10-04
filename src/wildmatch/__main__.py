"""``python -m wildmatch`` is the ``wildmatch`` command (used by sweep tasks)."""

import sys

from wildmatch.cli import main

if __name__ == "__main__":
    sys.exit(main())
