"""Top-level package for SOK: LDP Analysis."""
__version__ = "0.1.0"


def conf_rel_path():
    """Configure relative path imports for the experiments folder."""
    import sys
    from pathlib import Path

    parent_folder = str(Path("..").resolve())
    if parent_folder not in sys.path:
        sys.path.append(parent_folder)
