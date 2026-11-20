from dataclasses import dataclass
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(env_file: Path = REPO_ROOT / ".env") -> None:
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:  # empty values in .env.example mean "unset", not ""
            os.environ.setdefault(key.strip(), value)


load_dotenv()


def _env_path(name: str, default: str) -> Path:
    return REPO_ROOT / Path(os.environ.get(name, default)).expanduser()  # relative = repo root


# Config used when a script gets no --config: the final method (DEMO_CONFIG in .env overrides it)
DEFAULT_CONFIG = _env_path("DEMO_CONFIG", "configs/tracker/flatsam_final.py")


@dataclass
class PathConfig:
    POT_BASE: Path
    PT_BASE: Path
    PT_INITFIX_DIR: Path
    POT280_BASE: Path
    POIC_BASE: Path
    UCSB_BASE: Path
    MPOT_BASE: Path
    OUTPUT_DIRECTORY: Path
    VIS_DIRECTORY: Path


# Defaults match .env.example; every field can be overridden by the same-named var in .env
paths = PathConfig(
    POT_BASE=_env_path("POT_BASE", "data/POT-210/"),
    PT_BASE=_env_path("PT_BASE", "data/PlanarTrack/PlanarTrack-Data/"),
    PT_INITFIX_DIR=_env_path("PT_INITFIX_DIR", "data/PlanarTrack/PlanarTrack_init_reannotation/"),
    POT280_BASE=_env_path("POT280_BASE", "data/POT-280/"),
    POIC_BASE=_env_path("POIC_BASE", "data/POIC/"),
    UCSB_BASE=_env_path("UCSB_BASE", "data/UCSB/"),
    MPOT_BASE=_env_path("MPOT_BASE", "data/MPOT-3K/"),
    OUTPUT_DIRECTORY=_env_path("OUTPUT_DIRECTORY", "results/"),
    VIS_DIRECTORY=_env_path("VIS_DIRECTORY", "vis/"),
)
