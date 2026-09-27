from __future__ import annotations

import butterfly_saxs


def test_2d_tools_are_discoverable_public_exports():
    names = {
        "measure_point",
        "extract_line_profile",
        "measure_roi_orientation",
        "measure_azimuthal_profile",
        "fit_azimuthal_peaks",
        "analyze_density2d",
        "simulate_projected_density_fft",
        "export_lamellar_scattering",
        "inspect_legacy_saxs",
    }
    assert names <= set(butterfly_saxs.__all__)
    assert names <= set(dir(butterfly_saxs))
    assert all(callable(getattr(butterfly_saxs, name)) for name in names)
