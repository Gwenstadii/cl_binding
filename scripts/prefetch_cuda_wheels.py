"""Prepare checksum-verified CUDA wheels using the server's existing aria2c."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from email.parser import Parser
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import parse_qs, unquote, urljoin, urlsplit, urlunsplit
from urllib.request import urlopen

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
INDEX = "https://download.pytorch.org/whl/cu128/"
COMPATIBLE_TAGS = set(sys_tags())


def cached_wheel(name, version):
    cache = Path(os.environ.get("UV_CACHE_DIR", Path.home() / ".cache" / "uv")) / "archive-v0"
    if not cache.is_dir():
        return False
    for metadata_path in cache.glob("*/*.dist-info/METADATA"):
        message = Parser().parsestr(metadata_path.read_text(errors="replace"))
        if canonicalize_name(message.get("Name", "")) != canonicalize_name(name):
            continue
        candidate = message.get("Version", "")
        if candidate in SpecifierSet("==" + version) and (name != "torch" or candidate.endswith("+cu128")):
            return True
    return False


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "a" and "href" in attributes:
            self.links.append(attributes["href"])


def wheel_for(item):
    name, version = item
    page = urljoin(INDEX, canonicalize_name(name) + "/")
    parser = Links()
    parser.feed(urlopen(page, timeout=45).read().decode())
    candidates = []
    for link in parser.links:
        absolute = urljoin(page, link)
        parsed = urlsplit(absolute)
        filename = unquote(Path(parsed.path).name)
        try:
            distribution, candidate_version, _, tags = parse_wheel_filename(filename)
        except ValueError:
            continue
        if canonicalize_name(distribution) != canonicalize_name(name):
            continue
        if candidate_version not in SpecifierSet("==" + version) or not tags.intersection(COMPATIBLE_TAGS):
            continue
        allowed_hosts = {"download.pytorch.org", "download-r2.pytorch.org"}
        if name.startswith("nvidia-"):
            allowed_hosts.add("pypi.nvidia.com")
        if parsed.hostname not in allowed_hosts:
            raise RuntimeError(f"Unexpected wheel host for {name}: {parsed.hostname}")
        checksum = parse_qs(parsed.fragment).get("sha256", [None])[0]
        if not checksum or len(checksum) != 64:
            raise RuntimeError(f"No official SHA-256 for {filename}")
        candidates.append({"name": name, "version": str(candidate_version), "filename": filename, "url": urlunsplit(parsed._replace(fragment="")), "sha256": checksum})
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one compatible CUDA wheel for {name} {version}; got {len(candidates)}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Run aria2c; otherwise only prepare the plan")
    args = parser.parse_args()
    torch_wheel = wheel_for(("torch", "2.10.0"))
    metadata = Parser().parsestr(urlopen(torch_wheel["url"] + ".metadata", timeout=45).read().decode())
    small_or_cached = {"nvidia-cuda-runtime-cu12", "nvidia-cuda-cupti-cu12", "nvidia-nvtx-cu12", "nvidia-cufile-cu12", "nvidia-nvjitlink-cu12"}
    requested = []
    for entry in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(entry)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        if not (requirement.name.startswith("nvidia-") or requirement.name == "triton") or requirement.name in small_or_cached:
            continue
        pins = list(requirement.specifier)
        if len(pins) != 1 or pins[0].operator != "==":
            raise RuntimeError(f"Unpinned CUDA dependency: {entry}")
        if not cached_wheel(requirement.name, pins[0].version):
            requested.append((requirement.name, pins[0].version))
    with ThreadPoolExecutor(max_workers=4) as pool:
        wheels = [torch_wheel] + list(pool.map(wheel_for, requested))
    directory = ROOT / ".server" / "wheels"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(wheels, indent=2) + "\n")
    lines = []
    for wheel in wheels:
        lines.extend([wheel["url"], f"  dir={directory}", f"  out={wheel['filename']}", f"  checksum=sha-256={wheel['sha256']}"])
    input_file = directory / "aria2-input.txt"
    input_file.write_text("\n".join(lines) + "\n")
    print(json.dumps({"wheel_count": len(wheels), "directory": str(directory), "download_requested": args.download}), flush=True)
    if args.download:
        executable = shutil.which("aria2c")
        if not executable:
            raise RuntimeError("aria2c is not installed")
        subprocess.run([executable, "--continue=true", "--auto-file-renaming=false", "--allow-overwrite=false", "--check-integrity=true", "--max-connection-per-server=16", "--split=16", "--min-split-size=8M", "--max-concurrent-downloads=4", "--file-allocation=none", "--max-tries=5", "--retry-wait=5", "--timeout=60", "--connect-timeout=20", "--console-log-level=warn", "--summary-interval=30", "--download-result=hide", f"--input-file={input_file}"], check=True)


if __name__ == "__main__":
    main()
