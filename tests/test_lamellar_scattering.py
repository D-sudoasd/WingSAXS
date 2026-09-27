from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from butterfly_saxs.font_support import font_properties
from butterfly_saxs.lamellar_models import LamellarScene
from butterfly_saxs.lamellar_scattering import (
    export_lamellar_scattering,
    render_lamellar_scattering,
    simulate_projected_density_fft,
)
from butterfly_saxs.ui.qt_compat import QT_AVAILABLE


_SIGNS = np.asarray(
    [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ],
    dtype=float,
)


def _scene(
    *,
    angle_deg: float = 0.0,
    sizes: tuple[float, float, float] = (2.0, 1.5, 0.6),
    length_unit: str = "relative",
) -> LamellarScene:
    angle = np.deg2rad(angle_deg)
    rotation = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, np.cos(angle), -np.sin(angle)),
            (0.0, np.sin(angle), np.cos(angle)),
        ),
        dtype=float,
    )
    center = np.zeros((1, 3), dtype=float)
    size_array = np.asarray((sizes,), dtype=float)
    orientations = rotation[None, :, :]
    vertices = (
        (_SIGNS[None, :, :] * (size_array[:, None, :] / 2.0))
        @ np.swapaxes(orientations, 1, 2)
        + center[:, None, :]
    )
    return LamellarScene(
        centers=center,
        sizes=size_array,
        orientations=orientations,
        vertices=vertices,
        stack_ids=np.asarray([0]),
        branch_ids=np.asarray([0]),
        colors=np.asarray([[0.2, 0.45, 0.72, 0.82]]),
        bounds=np.asarray(
            (vertices.reshape(-1, 3).min(axis=0), vertices.reshape(-1, 3).max(axis=0))
        ),
        length_unit=length_unit,
        metadata={"status": "schematic", "flags": ["illustrative"]},
    )


def test_projection_integrates_finite_prism_volume_and_fft_axes() -> None:
    scene = _scene(angle_deg=37.0)
    result = simulate_projected_density_fft(scene, shape=(240, 256), margin_fraction=0.4)

    assert result.projected_density.shape == (240, 256)
    assert result.intensity.shape == (240, 256)
    assert result.qx.shape == (256,)
    assert result.qy.shape == (240,)
    assert np.isfinite(result.projected_density).all()
    assert np.isfinite(result.intensity).all()
    assert np.all(result.intensity >= 0.0)

    dx, dy = result.pixel_size
    volume_from_projection = float(np.sum(result.projected_density) * dx * dy)
    assert volume_from_projection == pytest.approx(2.0 * 1.5 * 0.6, rel=0.025)

    np.testing.assert_allclose(
        result.qx,
        np.fft.fftshift(2 * np.pi * np.fft.fftfreq(256, d=dx)),
    )
    np.testing.assert_allclose(
        result.qy,
        np.fft.fftshift(2 * np.pi * np.fft.fftfreq(240, d=dy)),
    )
    assert result.intensity[120, 128] <= result.intensity.max() * 1e-24
    assert result.diagnostics["nyquist_q_xy"] == pytest.approx(
        [np.pi / dx, np.pi / dy]
    )


def test_simulation_is_deterministic_and_keeps_relative_units() -> None:
    scene = _scene()
    scene_before = scene.to_mapping()
    first = simulate_projected_density_fft(scene, shape=(96, 112))
    second = simulate_projected_density_fft(scene, shape=(96, 112))

    np.testing.assert_array_equal(first.projected_density, second.projected_density)
    np.testing.assert_array_equal(first.intensity, second.intensity)
    assert first.q_unit == "relative⁻¹"
    assert "nm" not in first.density_unit
    assert first.metadata()["length_unit"] == "relative"
    assert first.metadata()["model_scope"] == "illustrative_projected_density_fft_only"
    assert first.metadata()["density_contrast"]["value"] == 1.0
    for name, expected in scene_before.items():
        actual = scene.to_mapping()[name]
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


def test_overlapping_slabs_add_projected_path_length() -> None:
    single = _scene()
    two_slabs = LamellarScene(
        centers=np.vstack((single.centers, single.centers)),
        sizes=np.vstack((single.sizes, single.sizes)),
        orientations=np.concatenate((single.orientations, single.orientations)),
        vertices=np.concatenate((single.vertices, single.vertices)),
        stack_ids=np.asarray([0, 1]),
        branch_ids=np.asarray([0, 0]),
        colors=np.concatenate((single.colors, single.colors)),
        bounds=single.bounds.copy(),
        length_unit=single.length_unit,
        metadata={"status": "schematic"},
    )
    one = simulate_projected_density_fft(single, shape=(80, 88))
    two = simulate_projected_density_fft(two_slabs, shape=(80, 88))

    np.testing.assert_allclose(two.projected_density, 2.0 * one.projected_density)
    np.testing.assert_allclose(two.intensity, 4.0 * one.intensity, rtol=1e-12, atol=1e-12)


def test_resolution_and_periodic_boundary_diagnostics_are_reported() -> None:
    result = simulate_projected_density_fft(
        _scene(sizes=(2.0, 1.5, 0.001)), shape=(16, 16), margin_fraction=0.0
    )
    warnings = result.diagnostics["warnings"]
    assert result.diagnostics["thinnest_slab_dimension_pixels"] < 2.0
    assert any("fewer than two" in warning for warning in warnings)
    assert any("periodic FFT boundary" in warning for warning in warnings)


def test_chinese_renderer_localizes_axes_and_density_units(tmp_path: Path) -> None:
    result = simulate_projected_density_fft(_scene(length_unit="nm"), shape=(64, 72))
    figure = render_lamellar_scattering(result, language="zh_CN")
    density_axis, intensity_axis = figure.axes[:2]
    assert density_axis.get_title(loc="left") == "投影密度"
    assert density_axis.get_xlabel() == "x（nm）"
    assert intensity_axis.get_title(loc="left") == "二维 FFT 功率"
    assert intensity_axis.get_xlabel() == "q_x（nm^-1）"
    assert figure.axes[2].get_ylabel() == "模型电子密度 × nm"
    assert figure.axes[3].get_ylabel() == "对数色标，I / max(I)"
    expected_font = font_properties(size=None, language="zh").get_file()
    assert expected_font
    chinese_artists = [
        density_axis._left_title,
        density_axis.xaxis.label,
        density_axis.yaxis.label,
        intensity_axis._left_title,
        intensity_axis.xaxis.label,
        intensity_axis.yaxis.label,
        figure.axes[2].yaxis.label,
        figure.axes[3].yaxis.label,
        figure._suptitle,
        *density_axis.get_xticklabels(),
        *density_axis.get_yticklabels(),
        *intensity_axis.get_xticklabels(),
        *intensity_axis.get_yticklabels(),
        *figure.axes[2].get_xticklabels(),
        *figure.axes[2].get_yticklabels(),
        *figure.axes[3].get_xticklabels(),
        *figure.axes[3].get_yticklabels(),
    ]
    wrong_fonts = [
        (artist.get_text(), artist.get_fontproperties().get_file())
        for artist in chinese_artists
        if artist.get_text()
        and artist.get_fontproperties().get_file() != expected_font
    ]
    assert not wrong_fonts, wrong_fonts
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        figure.savefig(tmp_path / "localized.png", dpi=40)
    assert not [warning for warning in observed if "Glyph" in str(warning.message)]


@pytest.mark.skipif(not QT_AVAILABLE, reason="desktop Qt runtime is unavailable")
def test_standalone_dialog_calculates_and_enables_export(qtbot) -> None:
    from butterfly_saxs.ui.lamellar_scattering_dialog import LamellarScatteringDialog

    dialog = LamellarScatteringDialog(_scene(length_unit="nm"), language="en")
    qtbot.addWidget(dialog)
    assert dialog._result is not None
    assert dialog.export_button.isEnabled()
    assert "q Nyquist" in dialog.status.text()


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((15, 32), "dimension"),
        ((64, 4096), "dimension"),
        ((2048, 2048), "too large"),
        ((32.5, 32), "integers"),
    ],
)
def test_grid_shape_is_bounded_and_validated(shape, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_projected_density_fft(_scene(), shape=shape)


def test_empty_scene_and_bad_geometry_fail_with_actionable_errors() -> None:
    empty = LamellarScene(
        centers=np.empty((0, 3)),
        sizes=np.empty((0, 3)),
        orientations=np.empty((0, 3, 3)),
        vertices=np.empty((0, 8, 3)),
        stack_ids=np.empty((0,), dtype=int),
        branch_ids=np.empty((0,), dtype=int),
        colors=np.empty((0, 4)),
        bounds=np.asarray(((-1, -1, -1), (1, 1, 1)), dtype=float),
        length_unit="relative",
        metadata={},
    )
    with pytest.raises(ValueError, match="contains no finite slabs"):
        simulate_projected_density_fft(empty)

    bad = _scene()
    bad.orientations[0] *= 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        simulate_projected_density_fft(bad)


def test_export_writes_strict_json_arrays_and_paired_plot_without_overwrite(
    tmp_path: Path,
) -> None:
    result = simulate_projected_density_fft(_scene(length_unit="nm"), shape=(80, 88))
    npz_path, json_path, png_path = export_lamellar_scattering(
        result, tmp_path / "forward-model.npz"
    )
    assert npz_path.exists() and json_path.exists() and png_path.exists()
    assert png_path.stat().st_size > 1000

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["model_scope"] == "illustrative_projected_density_fft_only"
    assert metadata["q_unit"] == "nm⁻¹"
    assert "not measured scattering" in metadata["interpretation"]
    assert metadata["assumptions"]
    with np.load(npz_path, allow_pickle=False) as archive:
        assert {
            "projected_density",
            "qx",
            "qy",
            "intensity",
            "intensity_normalized",
            "metadata_json",
        }.issubset(archive.files)
        embedded = json.loads(str(archive["metadata_json"].item()))
        assert embedded == metadata
        assert archive["projected_density"].shape == result.projected_density.shape

    preserved = npz_path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_lamellar_scattering(result, npz_path)
    assert npz_path.read_bytes() == preserved


def test_export_requires_existing_output_directory(tmp_path: Path) -> None:
    result = simulate_projected_density_fft(_scene(), shape=(32, 32))
    with pytest.raises(FileNotFoundError):
        export_lamellar_scattering(result, tmp_path / "missing" / "result.npz")
