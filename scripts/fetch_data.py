#!/usr/bin/env python
from pathlib import Path

from src.data.loader import ensure_brazil_combined_data


def main():
    path = ensure_brazil_combined_data()
    print(f"[data] Brazil combined data ready: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
