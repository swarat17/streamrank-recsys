"""
Top-level conftest.py — applies to all test suites.

Redirects MLflow to a temporary SQLite database so unit tests never depend on
the local mlruns/ directory structure or a running MLflow server.
"""

import pytest
import mlflow


@pytest.fixture(autouse=True, scope="session")
def _mlflow_tmp_tracking(tmp_path_factory):
    """Point MLflow at a fresh temp SQLite DB for the entire test session."""
    db = tmp_path_factory.mktemp("mlflow") / "test_mlruns.db"
    uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(uri)
    yield
    mlflow.set_tracking_uri("mlruns")  # restore default after session
