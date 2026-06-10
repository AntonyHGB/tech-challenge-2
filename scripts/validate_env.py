"""Environment validation script for the recommendation system.

Run with ``poetry run python scripts/validate_env.py`` (or ``python`` directly).
It verifies the interpreter version, that every required third-party package
imports, and that the typed settings load from the environment. The script
exits non-zero on the first failing category so it can gate a fresh install or
a CI step, fulfilling the Stage 2 "verify clean install" requirement.
"""

from __future__ import annotations

import importlib
import sys
from importlib import metadata
from pathlib import Path

# Make ``recsys`` importable when running the script straight from a checkout
# (i.e. before ``poetry install`` makes the package available on the path).
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MIN_PYTHON: tuple[int, int] = (3, 12)

# Importable module name -> distribution name on PyPI.
REQUIRED_PACKAGES: dict[str, str] = {
    "torch": "torch",
    "sklearn": "scikit-learn",
    "mlflow": "mlflow",
    "dvc": "dvc",
    "numpy": "numpy",
    "pandas": "pandas",
    "yaml": "pyyaml",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
}


def report(ok: bool, message: str) -> None:
    """Print a single check result with an aligned status marker.

    Args:
        ok: Whether the check passed.
        message: Human-readable description of the check.
    """
    marker = "OK  " if ok else "FAIL"
    print(f"[{marker}] {message}")


def check_python_version() -> bool:
    """Verify the running interpreter meets the minimum supported version.

    Returns:
        ``True`` if the Python version is supported.
    """
    current = sys.version_info[:2]
    ok = current >= MIN_PYTHON
    want = ".".join(map(str, MIN_PYTHON))
    have = ".".join(map(str, current))
    report(ok, f"Python {have} (requires >= {want})")
    return ok


def _probe_package(module_name: str, dist_name: str) -> tuple[bool, str]:
    """Import a package and look up its installed distribution version.

    Args:
        module_name: Importable module name (e.g. ``"sklearn"``).
        dist_name: Distribution name on PyPI (e.g. ``"scikit-learn"``).

    Returns:
        A ``(success, message)`` pair describing the probe result.
    """
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False, f"{dist_name}: not importable (run 'poetry install')"
    try:
        version = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        version = "unknown"
    return True, f"{dist_name} {version}"


def check_packages() -> bool:
    """Check that every required package imports and report its version.

    Returns:
        ``True`` if all required packages import successfully.
    """
    all_ok = True
    for module_name, dist_name in REQUIRED_PACKAGES.items():
        ok, detail = _probe_package(module_name, dist_name)
        report(ok, detail)
        all_ok = all_ok and ok
    return all_ok


def check_settings() -> bool:
    """Load the typed settings to confirm configuration wiring works.

    Returns:
        ``True`` if the settings object loads without error.
    """
    try:
        from recsys.config import get_settings

        settings = get_settings()
    except Exception as exc:
        report(False, f"settings failed to load: {exc}")
        return False
    report(
        True,
        f"settings loaded (seed={settings.random_seed}, "
        f"mlflow='{settings.mlflow_experiment_name}')",
    )
    return True


def main() -> int:
    """Run every environment check and summarise the outcome.

    Returns:
        ``0`` if all checks passed, ``1`` otherwise.
    """
    print("Validating environment for ecommerce-recsys...\n")
    results = [check_python_version(), check_packages(), check_settings()]
    ok = all(results)
    print()
    print("Environment OK." if ok else "Environment validation FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
