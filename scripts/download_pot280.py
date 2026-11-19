#!/usr/bin/env python3
"""Download and extract POT-280 one Google Drive object archive at a time."""

import argparse
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import cv2


DATASET_PAGE = "https://liangp.github.io/data/pot280/"
ANNOTATION_URL = urllib.parse.urljoin(DATASET_PAGE, "annotation.zip")
OBJECT_LINK_RE = re.compile(
    r'href="https://drive\.google\.com/open\?id=([^"]+)"[^>]*>([^<]+)</a>',
    re.IGNORECASE,
)


def object_links() -> Dict[str, str]:
    """Read current object archive IDs from the benchmark page."""
    with urllib.request.urlopen(DATASET_PAGE) as response:
        page = response.read().decode("utf-8")
    links = OBJECT_LINK_RE.findall(page)
    if len(links) != 40:
        raise RuntimeError(f"Expected 40 POT-280 object links, found {len(links)}")
    return {f"V{index:02d}": file_id for index, (file_id, _) in enumerate(links, start=1)}


def drive_download_url(file_id: str) -> str:
    """Return a Google Drive URL that accepts the large-file confirmation."""
    first_url = (
        "https://drive.google.com/uc?"
        + urllib.parse.urlencode({"export": "download", "id": file_id})
    )
    request = urllib.request.Request(first_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        body = response.read().decode("utf-8", errors="replace")
    match = re.search(r'name="confirm" value="([^"]+)"', body)
    uuid_match = re.search(r'name="uuid" value="([^"]+)"', body)
    params = {"export": "download", "id": file_id, "confirm": match.group(1) if match else "t"}
    if uuid_match:
        params["uuid"] = uuid_match.group(1)
    return "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(params)


def download_file(url: str, destination: Path) -> None:
    """Download *url* to *destination*, refusing an HTML error response."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        content_type = response.headers.get_content_type()
        if content_type == "text/html":
            raise RuntimeError(f"Google Drive returned an HTML error page for {url}")
        shutil.copyfileobj(response, output)


def safe_extract(archive: Path, destination: Path) -> Iterable[Path]:
    """Extract an archive while rejecting paths outside *destination*."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        members = []
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            zip_file.extract(member, destination)
            members.append(target)
    return members


def extract_videos_to_frames(
    archive_root: Path, dataset_root: Path, selected_sequences: Optional[Set[str]] = None
) -> None:
    """Convert object archive AVI files into numbered JPEG frame directories."""
    for video_path in sorted(archive_root.rglob("*.avi")):
        if video_path.name.startswith("._") or "__MACOSX" in video_path.parts:
            continue
        sequence = video_path.stem
        if selected_sequences is not None and sequence not in selected_sequences:
            continue
        frame_dir = dataset_root / "extracted" / "images" / sequence
        frame_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(frame_dir.glob("*.jpg"))
        if len(existing) == 501:
            continue
        for frame in existing:
            frame.unlink()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frame_index = 0
        try:
            while True:
                success, image = capture.read()
                if not success:
                    break
                output_path = frame_dir / f"{frame_index:05d}.jpg"
                if not cv2.imwrite(str(output_path), image):
                    raise RuntimeError(f"Could not write frame: {output_path}")
                frame_index += 1
        finally:
            capture.release()
        if frame_index != 501:
            raise RuntimeError(f"{video_path} produced {frame_index} frames, expected 501")


def download_object(
    object_name: str,
    file_id: str,
    dataset_root: Path,
    selected_sequences: Optional[Set[str]] = None,
) -> None:
    """Download, extract, and frame one POT-280 object archive."""
    archive_path = dataset_root / "archives" / f"{object_name}.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        with tempfile.NamedTemporaryFile(
            dir=archive_path.parent, prefix=f"{object_name}.", suffix=".part", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            download_file(drive_download_url(file_id), temporary_path)
            temporary_path.replace(archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{object_name}-", dir=dataset_root) as temporary_dir:
        extracted_root = Path(temporary_dir)
        safe_extract(archive_path, extracted_root)
        extract_videos_to_frames(extracted_root, dataset_root, selected_sequences)


def download_annotations(dataset_root: Path) -> None:
    """Download and extract the shared ground-truth archive once."""
    annotation_dir = dataset_root / "annotation"
    if next(annotation_dir.glob("*_gt_points.txt"), None) is not None:
        return
    archive_path = dataset_root / "archives" / "annotation.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        download_file(ANNOTATION_URL, archive_path)
    safe_extract(archive_path, dataset_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="POT-280 dataset root")
    parser.add_argument("--object", help="Object name, e.g. V01; use all for every object")
    parser.add_argument(
        "--sequences",
        nargs="+",
        metavar="SEQ",
        help="Only extract named sequences, e.g. V01_1; downloads containing objects still apply",
    )
    args = parser.parse_args()
    links = object_links()
    if args.object is None and not args.sequences:
        print("Available objects: " + " ".join(sorted(links)))
        return
    if args.object == "all":
        names = sorted(links)
    elif args.object is None:
        names = sorted({sequence.split("_", 1)[0] for sequence in args.sequences})
    else:
        names = [args.object]
    unknown = [name for name in names if name not in links]
    if unknown:
        raise ValueError(f"Unknown POT-280 object(s): {', '.join(unknown)}")
    args.root.mkdir(parents=True, exist_ok=True)
    download_annotations(args.root)
    for name in names:
        print(f"Downloading {name}")
        download_object(
            name,
            links[name],
            args.root,
            selected_sequences=set(args.sequences) if args.sequences else None,
        )


if __name__ == "__main__":
    main()
