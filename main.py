from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gui import QuickDashCamApp


def main() -> None:
    app = QuickDashCamApp()
    app.mainloop()


if __name__ == "__main__":
    main()
