# SPDX-License-Identifier: GPL-3.0-or-later
"""`src/arjentic/` must stay a PEP 420 implicit namespace package.

An `__init__.py` here is the canonical way to silently break the namespace
package, and it fails late (at import time in a fresh environment), not at
install time -- hence a standing test rather than trusting review.
"""

from pathlib import Path


def test_arjentic_namespace_has_no_init() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    init_file = repo_root / "src" / "arjentic" / "__init__.py"
    assert not init_file.exists(), (
        "src/arjentic/__init__.py must not exist -- it would turn "
        "'arjentic' into a regular package and break namespace packaging."
    )
