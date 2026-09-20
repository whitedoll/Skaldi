"""python -m skaldi 로도 실행할 수 있게 한다."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
