"""Download the UCI Online Retail dataset into Data/external/uci_online_retail/ (ignored by Git).

Source:  https://archive.ics.uci.edu/dataset/352/online+retail
Citation: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.
          https://doi.org/10.24432/C5BW33
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)

The script downloads the official ZIP (about 23 MB), extracts "Online Retail.xlsx", converts it to
online_retail.csv and writes download_manifest.json with the retrieval time and SHA-256 checksums.
A ZIP that is already there is reused; pass --force to download it again. Run from the project root:
    python scripts/download_uci_online_retail.py
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market_basket.uci import DATA_DIR, DOWNLOAD_URL, DatasetError, download_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"where to store the files (default {DATA_DIR})")
    parser.add_argument("--url", default=DOWNLOAD_URL, help="download URL (default: the official UCI ZIP)")
    parser.add_argument("--force", action="store_true", help="download the ZIP again even if it is already there")
    args = parser.parse_args(argv)
    try:
        manifest = download_dataset(args.data_dir, args.url, force=args.force)
    except (DatasetError, OSError, urllib.error.URLError) as error:
        print(f"error: {error}", file=sys.stderr)
        print(f"If the UCI site is unavailable, download the ZIP manually from {DOWNLOAD_URL} into "
              f"{args.data_dir} and run this script again (it reuses a ZIP that is already there).", file=sys.stderr)
        return 1
    for label, entry in manifest["files"].items():
        print(f"  {label:<12} {entry['name']:<22} {entry['bytes']:>12,} bytes  sha256 {entry['sha256']}")
    print(f"Cite as: {manifest['citation']} Licence: {manifest['licence']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
