"""Fetch and hash-verify the exact primary count and metadata sources."""

import argparse
import json
import urllib.request
from pathlib import Path

import truststore
from wessels_prepare import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/wessels-raw"))
    args = parser.parse_args()
    truststore.inject_into_ssl()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("data/wessels-pseudobulk.manifest.json").read_text())
    for key, hash_key in [
        ("source", "archive_sha256"),
        ("metadata_source", "metadata_sha256"),
    ]:
        url = manifest[key]
        path = args.output / url.rsplit("/", 1)[1]
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                temporary.open("wb") as handle,
            ):
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(block)
            if sha256(temporary) != manifest[hash_key]:
                raise ValueError(
                    "Source differs from recorded snapshot; inspect before updating the experiment"
                )
            temporary.replace(path)
        if sha256(path) != manifest[hash_key]:
            raise ValueError("Cached source checksum mismatch")
        print(path, "verified")


if __name__ == "__main__":
    main()
