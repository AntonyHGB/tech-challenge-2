"""Validação de ambiente do sistema de recomendação.

Execute com ``poetry run python scripts/validate_env.py``. O script confere a
versão do interpretador, se cada dependência obrigatória importa e se as
configurações tipadas carregam do ambiente. Sai com código diferente de zero na
primeira categoria que falhar, servindo de portão para uma instalação limpa.
"""

from __future__ import annotations

import importlib
import sys
from importlib import metadata
from pathlib import Path

# Torna ``recsys`` importável ao rodar direto do checkout, ou seja, antes de o
# ``poetry install`` deixar o pacote disponível no path.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MIN_PYTHON: tuple[int, int] = (3, 12)

# Nome do módulo importável -> nome da distribuição no PyPI.
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
    """Imprime o resultado de uma verificação com marcador alinhado.

    Args:
        ok: Se a verificação passou.
        message: Descrição legível da verificação.
    """
    marker = "OK  " if ok else "FALHA"
    print(f"[{marker}] {message}")


def check_python_version() -> bool:
    """Confere se o interpretador atende à versão mínima suportada."""
    current = sys.version_info[:2]
    ok = current >= MIN_PYTHON
    want = ".".join(map(str, MIN_PYTHON))
    have = ".".join(map(str, current))
    report(ok, f"Python {have} (exige >= {want})")
    return ok


def check_packages() -> bool:
    """Confere se cada pacote obrigatório importa e mostra a versão."""
    all_ok = True
    for module_name, dist_name in REQUIRED_PACKAGES.items():
        ok, detail = _probe_package(module_name, dist_name)
        report(ok, detail)
        all_ok = all_ok and ok
    return all_ok


def check_settings() -> bool:
    """Carrega as configurações tipadas para conferir a ligação com o ``.env``."""
    try:
        from recsys.config import get_settings

        settings = get_settings()
    except Exception as exc:
        report(False, f"falha ao carregar as configurações: {exc}")
        return False
    report(
        True,
        f"configurações carregadas (seed={settings.random_seed}, "
        f"mlflow='{settings.mlflow_experiment_name}')",
    )
    return True


def main() -> int:
    """Roda todas as verificações e resume o resultado.

    Returns:
        ``0`` se tudo passou, ``1`` caso contrário.
    """
    print("Validando o ambiente do ecommerce-recsys...\n")
    results = [check_python_version(), check_packages(), check_settings()]
    ok = all(results)
    print()
    print("Ambiente OK." if ok else "Validação do ambiente FALHOU.")
    return 0 if ok else 1


def _probe_package(module_name: str, dist_name: str) -> tuple[bool, str]:
    """Importa um pacote e devolve ``(sucesso, mensagem)`` com a versão."""
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False, f"{dist_name}: não importa (rode 'poetry install')"
    try:
        version = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        version = "desconhecida"
    return True, f"{dist_name} {version}"


if __name__ == "__main__":
    raise SystemExit(main())
