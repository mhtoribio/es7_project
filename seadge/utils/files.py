import fnmatch
from pathlib import Path

def files_in_path_recursive(path: Path, glob: str) -> list[Path]:
    """
    Returns a list of Path objects of wav files in a path (recursive dir search or single file).

    - If `path` is a file: returns [path] if it matches `pattern` (case-insensitive), else [].
    - If `path` is a directory: recursively searches and returns files matching `pattern`
      (case-insensitive) using rglob.
    - If `path` does not exist: returns [].
    - Output is sorted by path string for determinism.
    """
    try:
        p = Path(path).expanduser()
    except Exception:
        p = Path(path)

    if not p.exists():
        return []

    pat = glob.lower()

    def _matches(name: str) -> bool:
        # case-insensitive fnmatch
        return fnmatch.fnmatchcase(name.lower(), pat)

    results: list[Path] = []

    if p.is_file():
        if _matches(p.name):
            results = [p]
    elif p.is_dir():
        # recursive search; filter manually for case-insensitive matching
        for f in p.rglob("*"):
            try:
                if f.is_file() and _matches(f.name):
                    results.append(f)
            except PermissionError:
                # Skip unreadable entries
                continue
    else:
        return []

    results.sort(key=lambda x: str(x).lower())
    return results
