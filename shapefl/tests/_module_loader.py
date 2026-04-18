from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def load_utils_module(module_name: str):
    _ensure_package("shapefl", ROOT / "shapefl")
    _ensure_package("shapefl.utils", ROOT / "shapefl" / "utils")
    full_name = f"shapefl.utils.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    module_path = ROOT / "shapefl" / "utils" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {full_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module
