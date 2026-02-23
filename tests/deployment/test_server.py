"""Tests for the inference server.

Uses FastAPI TestClient with mocked model to verify all endpoints behave
correctly without needing a trained model.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prospect_rl.config import NUM_ORE_TYPES, Action
from prospect_rl.deployment.inference_server import _WINDOW_VOLUME

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_act_payload() -> dict:
    """Return a well-formed /act request body."""
    return {
        "position": [10.0, 8.0, 10.0],
        "facing": 0,
        "fuel": 500,
        "inventory": {"coal": 3, "iron": 1},
        "nearby_blocks": ["stone"] * _WINDOW_VOLUME,
        "preference": [0.0] * NUM_ORE_TYPES,
        "world_size": [64, 128, 64],
        "max_fuel": 1000,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_model():
    """Create a mock MaskablePPO model that returns action 0 (forward)."""
    model = MagicMock()
    model.predict.return_value = (np.int64(0), None)
    return model


@pytest.fixture()
def client(mock_model):
    """FastAPI TestClient with the model mocked in.

    The lifespan handler runs on TestClient enter and will fail to load the
    real model (no file on disk).  We inject the mock model *after* the
    lifespan runs so the endpoints see it.
    """
    import prospect_rl.deployment.inference_server as srv

    with TestClient(srv.app, raise_server_exceptions=False) as c:
        # Override the globals that the lifespan set to None
        srv._model = mock_model
        srv._vec_normalize = None
        yield c

    srv._model = None
    srv._vec_normalize = None


@pytest.fixture()
def client_no_model():
    """FastAPI TestClient with no model loaded (503 scenario)."""
    import prospect_rl.deployment.inference_server as srv

    with TestClient(srv.app, raise_server_exceptions=False) as c:
        # Ensure model stays None (lifespan already failed to load)
        srv._model = None
        srv._vec_normalize = None
        yield c

    srv._model = None
    srv._vec_normalize = None


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_model_not_loaded(
        self, client_no_model: TestClient,
    ) -> None:
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# Act endpoint
# ---------------------------------------------------------------------------

class TestAct:
    def test_act_returns_valid_action(self, client: TestClient) -> None:
        payload = _valid_act_payload()
        resp = client.post("/act", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["action"] <= 8
        assert data["action_name"] in [a.name.lower() for a in Action]
        assert 0.0 <= data["confidence"] <= 1.0

    def test_act_returns_503_without_model(
        self, client_no_model: TestClient,
    ) -> None:
        payload = _valid_act_payload()
        resp = client_no_model.post("/act", json=payload)
        assert resp.status_code == 503

    def test_act_invalid_input_returns_422(self, client: TestClient) -> None:
        # Missing required fields
        resp = client.post("/act", json={"position": [1, 2]})
        assert resp.status_code == 422

    def test_act_wrong_nearby_blocks_length(self, client: TestClient) -> None:
        payload = _valid_act_payload()
        payload["nearby_blocks"] = ["stone"] * 10  # wrong length
        resp = client.post("/act", json=payload)
        assert resp.status_code == 422

    def test_act_wrong_preference_length(self, client: TestClient) -> None:
        payload = _valid_act_payload()
        payload["preference"] = [1.0, 0.0]  # wrong length
        resp = client.post("/act", json=payload)
        assert resp.status_code == 422

    def test_act_calls_model_predict(
        self, client: TestClient, mock_model: MagicMock,
    ) -> None:
        payload = _valid_act_payload()
        resp = client.post("/act", json=payload)
        assert resp.status_code == 200
        mock_model.predict.assert_called_once()


# ---------------------------------------------------------------------------
# Preference endpoint
# ---------------------------------------------------------------------------

class TestPreference:
    def test_preference_diamond(self, client: TestClient) -> None:
        resp = client.post("/preference", json={"target": "diamond"})
        assert resp.status_code == 200
        pref = resp.json()["preference"]
        assert len(pref) == NUM_ORE_TYPES
        assert pref[3] == 1.0  # diamond is index 3
        assert sum(pref) == 1.0

    def test_preference_coal(self, client: TestClient) -> None:
        resp = client.post("/preference", json={"target": "coal"})
        assert resp.status_code == 200
        pref = resp.json()["preference"]
        assert pref[0] == 1.0

    def test_preference_iron(self, client: TestClient) -> None:
        resp = client.post("/preference", json={"target": "iron"})
        assert resp.status_code == 200
        pref = resp.json()["preference"]
        assert pref[1] == 1.0

    def test_preference_copper(self, client: TestClient) -> None:
        resp = client.post("/preference", json={"target": "copper"})
        assert resp.status_code == 200
        pref = resp.json()["preference"]
        assert pref[7] == 1.0  # copper is index 7

    def test_preference_unknown_ore(self, client: TestClient) -> None:
        resp = client.post("/preference", json={"target": "unobtainium"})
        assert resp.status_code == 400

    def test_preference_all_ores(self, client: TestClient) -> None:
        """Every known ore name should produce a valid one-hot vector."""
        ore_names = [
            "coal", "iron", "gold", "diamond",
            "redstone", "emerald", "lapis", "copper",
        ]
        for i, name in enumerate(ore_names):
            resp = client.post("/preference", json={"target": name})
            assert resp.status_code == 200, f"Failed for {name}"
            pref = resp.json()["preference"]
            assert pref[i] == 1.0, f"Wrong index for {name}"


# ---------------------------------------------------------------------------
# Load test (Task 8.4) -- 100 sequential requests
# ---------------------------------------------------------------------------

class TestLoadSequential:
    def test_100_sequential_requests(self, client: TestClient) -> None:
        """Send 100 sequential /act requests to verify stability."""
        payload = _valid_act_payload()
        for i in range(100):
            resp = client.post("/act", json=payload)
            assert resp.status_code == 200, (
                f"Request {i} failed with {resp.status_code}"
            )
            data = resp.json()
            assert 0 <= data["action"] <= 8
