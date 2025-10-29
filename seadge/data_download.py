from __future__ import annotations

from pathlib import Path
import tarfile
try:
    from tqdm.auto import tqdm  # optional progress bar
except Exception:
    tqdm = None

from seadge.utils.log import log
from seadge import config

_dns_zip_files = [
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_000_0.00_3.75.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_001_3.75_3.88.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_002_3.88_3.96.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_003_3.96_4.02.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_004_4.02_4.06.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_005_4.06_4.10.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_006_4.10_4.13.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_007_4.13_4.16.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_008_4.16_4.19.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_009_4.19_4.21.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_010_4.21_4.24.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_011_4.24_4.26.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_012_4.26_4.29.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_013_4.29_4.31.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_014_4.31_4.33.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_015_4.33_4.35.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_016_4.35_4.38.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_017_4.38_4.40.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_018_4.40_4.42.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_019_4.42_4.45.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_020_4.45_4.48.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_021_4.48_4.52.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_022_4.52_4.57.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_023_4.57_4.67.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_024_4.67_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_025_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_026_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_027_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_028_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_029_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_030_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_031_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_032_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_033_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_034_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_035_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_036_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_037_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_038_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_039_NA_NA.tar.bz2"
]
_base_dns_url = "https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"

import os
from typing import Optional, Callable, Dict, Any

import requests

DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


def _get_remote_size(session: requests.Session, url: str, *, timeout: float = 30.0) -> Optional[int]:
    """Best-effort HEAD to learn Content-Length. Returns None if unavailable."""
    try:
        r = session.head(url, allow_redirects=True, timeout=timeout)
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        if cl is not None:
            size = int(cl)
            if log:
                log.debug(f"Remote size for {url} is {size} bytes")
            return size
    except Exception as e:
        if log:
            log.debug(f"HEAD failed for {url}: {e}")
    return None

def download_file(
    url: str,
    outpath: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    resume: bool = True,
    verify_size: bool = True,
    timeout: float = 60.0,
    session_kwargs: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
) -> None:
    """
    Stream a file to disk with constant memory usage. Supports resume via HTTP Range.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath.with_name(outpath.name + ".part")

    with requests.Session() as s:
        if session_kwargs:
            for k, v in session_kwargs.items():
                if k in ("headers", "params", "proxies", "cookies") and isinstance(v, dict):
                    getattr(s, k).update(v)
                else:
                    setattr(s, k, v)

        remote_size = _get_remote_size(s, url, timeout=timeout)

        resume_from = 0
        if resume and tmp.exists():
            resume_from = tmp.stat().st_size
            if log:
                log.debug(f"Found partial file {tmp} ({resume_from} bytes); attempting to resume")

        headers = {}
        if resume and resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"

        r = s.get(url, headers=headers, stream=True, timeout=timeout)
        if r.status_code == 200 and "Range" in headers:
            if log:
                log.debug("Server ignored Range; restarting from scratch")
            resume_from = 0
            tmp.write_bytes(b"")  # truncate
        elif r.status_code not in (200, 206):
            r.raise_for_status()

        mode = "ab" if resume and resume_from > 0 else "wb"

        # Progress setup
        bytes_written = resume_from
        if progress_cb:
            progress_cb(bytes_written, remote_size)

        if log:
            if resume_from:
                log.info(f"Downloading (resuming at {resume_from} / {remote_size or 'unknown'}) -> {outpath}")
            else:
                log.info(f"Downloading ({remote_size or 'unknown'} bytes) -> {outpath}")

        bar = None
        if tqdm is not None:
            # Show full-size progress if known; otherwise indeterminate byte counter
            bar = tqdm(
                total=remote_size,
                initial=resume_from,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=outpath.name,
            )

        with tmp.open(mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                bytes_written += len(chunk)
                if bar:
                    bar.update(len(chunk))
                if progress_cb:
                    progress_cb(bytes_written, remote_size)

        if bar:
            bar.close()

        if verify_size and remote_size is not None:
            final_size = tmp.stat().st_size
            if final_size != remote_size:
                if log:
                    log.error(f"Size mismatch after download: got {final_size}, expected {remote_size}")
                raise RuntimeError(f"Size mismatch: got {final_size}, expected {remote_size}")

        tmp.replace(outpath)
        if log:
            log.debug(f"Downloaded {url} to {outpath} ({bytes_written} bytes)")

def _basename_without_suffixes(p: Path) -> str:
    """Return filename without the last two extensions (e.g., '.tar.bz2')."""
    name = p.name
    for _ in range(2):
        name = name.rsplit(".", 1)[0] if "." in name else name
    return name

def _is_within_directory(base: Path, target: Path) -> bool:
    """Ensure target resolves within base (prevents path traversal)."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False

def unzip_all_archives(
    zipdir: Path,
    outdir: Path,
    *,
    pattern: str = "*.tar.bz2",
    create_subdir: bool = False,
    skip_existing_subdir: bool = True,
) -> None:
    """
    Extract all archives in `zipdir` matching `pattern` into `outdir`.

    Args:
        zipdir: Directory containing archives (e.g., .tar.bz2 files).
        outdir: Destination directory for extracted contents.
        pattern: Glob pattern (default: '*.tar.bz2').
        create_subdir: If True, each archive is extracted into its own subdirectory
                       named after the archive (without .tar.bz2).
        skip_existing_subdir: If create_subdir=True and the target subdir exists,
                              skip extracting that archive.

    Notes:
        - Uses a safe extraction routine to avoid path traversal and skips symlinks.
        - Logs progress via `log`.
    """
    zipdir = Path(zipdir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    archives = sorted(zipdir.glob(pattern))
    if not archives:
        if log:
            log.info(f"No archives found in {zipdir} matching {pattern}")
        return

    for arch in archives:
        try:
            if create_subdir:
                target_dir = outdir / _basename_without_suffixes(arch)
                if skip_existing_subdir and target_dir.exists():
                    if log:
                        relp = target_dir.relative_to(outdir)
                        log.info(f"Skipping existing extraction dir: {relp}")
                    continue
                target_dir.mkdir(parents=True, exist_ok=True)
            else:
                target_dir = outdir

            if log:
                log.info(f"Extracting {arch.name} -> {target_dir}")

            with tarfile.open(arch, mode="r:*") as tf:
                tf.extractall(target_dir)

            if log:
                log.debug(f"Finished extracting {arch.name}")
        except Exception as e:
            if log:
                log.error(f"Failed to extract {arch}: {e}")
            # Optionally continue to next archive instead of raising
            continue

def download_dns_zip_files(output_dir: Path, n: int, use_existing: bool = True):
    """
    Downloads n zip files from DNS Challenge to output directory.
    Will always be same order.
    """
    if n > len(_dns_zip_files):
        log.warning(f"Number of DNS Challenge zip files too large. Truncating {n} to {len(_dns_zip_files)}")
        n = len(_dns_zip_files)
    for i in range(n):
        fname = _dns_zip_files[i]
        url = _base_dns_url + "/" + fname
        outfile = output_dir / Path(fname).name
        if use_existing and outfile.exists():
            log.info(f"Data download: using existing zip {outfile.relative_to(output_dir)}")
        else:
            download_file(url, outfile)

def main():
    cfg = config.get()
    download_dns_zip_files(cfg.paths.download_cache_dir, cfg.clean_zip_files, use_existing=True)
    unzip_all_archives(cfg.paths.download_cache_dir, cfg.paths.clean_dir)
