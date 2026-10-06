import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from momentum_scanner.run_premarket_scan import main

if __name__ == "__main__":
    main()
