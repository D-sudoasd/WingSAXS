from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from butterfly_saxs import geometry, service as service_module
from butterfly_saxs.io import LoadedImage
from butterfly_saxs.service import ButterflyAnalysisService


def test_failed_candidate_calibration_keeps_document_and_cache_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (4, 5)

    class Integrator:
        def __init__(self, q_value: float) -> None:
            self.q_value = q_value

    integrators = {
        "old.poni": Integrator(10.0),
        "new.poni": Integrator(100.0),
    }

    def load_poni(path: str | Path) -> Integrator:
        return integrators[str(path)]

    def build_geometry(image_shape: tuple[int, int], poni: Integrator) -> SimpleNamespace:
        q = np.full(image_shape, poni.q_value, dtype=float)
        qx = q.copy()
        qy = np.zeros(image_shape, dtype=float)
        return SimpleNamespace(
            q_nm_inv=q,
            qx_nm_inv=qx,
            qy_nm_inv=qy,
            chi_rad=np.zeros(image_shape, dtype=float),
            valid_mask=np.ones(image_shape, dtype=bool),
            q_unit="nm^-1",
            fingerprint=f"q={poni.q_value}",
            metadata={"q_unit": "nm^-1"},
        )

    def read_image(path: str | Path, **_kwargs: object) -> LoadedImage:
        return LoadedImage(
            np.full(shape, 2.0, dtype=float),
            source=Path(path),
        )

    monkeypatch.setattr(geometry, "load_poni", load_poni)
    monkeypatch.setattr(service_module, "build_geometry", build_geometry)
    monkeypatch.setattr(service_module, "read_image", read_image)

    service = ButterflyAnalysisService()
    service.set_observed(np.ones(shape, dtype=float))
    service.set_poni("old.poni")
    previous_loaded = service._loaded
    previous_qmap = service.qmap
    assert service.poni_path == "old.poni"

    with pytest.raises(ValueError, match="external_mask shape"):
        service.load_image(
            "candidate.npy",
            poni="new.poni",
            external_mask=np.ones((2, 2), dtype=bool),
        )

    assert service.poni_path == "old.poni"
    assert service._poni is integrators["old.poni"]
    assert service._loaded is previous_loaded
    assert service.qmap is previous_qmap

    service.load_image("next.npy")

    assert service.poni_path == "old.poni"
    assert service._poni is integrators["old.poni"]
    np.testing.assert_array_equal(
        service.qmap.q_nm_inv,
        np.full(shape, integrators["old.poni"].q_value),
    )
