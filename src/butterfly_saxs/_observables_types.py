"""Core dataclasses, mapping adapters, and reciprocal-unit/grid helpers for observables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._primitives import (
    normalize_q_unit as _normalized_q_unit,
    q_scale_to_nm_inv as _q_to_nm_inverse_scale,
)
from .models import RidgePoint as _SharedRidgePoint


APPARENT_FLAGS = (
    "apparent_geometry_only",
    "nonunique_inverse_problem",
)
# Relative tolerance for treating a fitted q-space ellipse as origin-centred
# before deriving the optional physical spacing proxies.
_SPACING_CENTER_REL_TOL = 1e-8


class _MappingResult:
    """Small common adapter: results work as both objects and mappings."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> tuple[str, ...]:
        return tuple(self.__dataclass_fields__)  # type: ignore[attr-defined]

    def as_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.keys()}


@dataclass
class AngularSpectrum(_MappingResult):
    """Azimuthal intensity profile in a q window.

    ``angle`` is in radians and is centred on ``[-pi, pi)``.  ``coverage`` is
    the fraction of finite candidate pixels in each angular bin that survived
    the detector mask; it is therefore useful for deciding whether an
    apparent lobe is supported by data rather than by an empty sector.
    """

    angle: np.ndarray
    intensity: np.ndarray
    counts: np.ndarray
    candidate_counts: np.ndarray
    coverage: np.ndarray
    q_min: float
    q_max: float
    q_center: float
    statistic: str = "mean"
    flags: tuple[str, ...] = APPARENT_FLAGS
    q_unit: str = "unknown"

    @property
    def azimuth(self) -> np.ndarray:
        return self.angle

    @property
    def profile(self) -> np.ndarray:
        return self.intensity

    @property
    def global_coverage(self) -> float:
        denom = float(np.sum(self.candidate_counts))
        return float(np.sum(self.counts) / denom) if denom else 0.0


@dataclass
class LobeMetrics(_MappingResult):
    """One measured angular lobe."""

    angle: float
    intensity: float
    baseline: float
    snr: float
    fwhm: float
    area: float
    index: int
    coverage: float
    n_pixels: int
    valid: bool = True
    flags: tuple[str, ...] = APPARENT_FLAGS
    reason: str = "accepted"
    refinement: str = "quadratic"

    @property
    def azimuth(self) -> float:
        return self.angle

    @property
    def azimuthal_fwhm(self) -> float:
        return self.fwhm

    @property
    def fwhm_deg(self) -> float:
        """Full width at half maximum in display-friendly degrees."""

        return float(np.degrees(self.fwhm)) if np.isfinite(self.fwhm) else float("nan")

    def keys(self) -> tuple[str, ...]:
        return super().keys() + ("fwhm_deg",)


@dataclass
class RadialProfile(_MappingResult):
    """Radial profile for one *observed* angular sector."""

    angle: float
    q: np.ndarray
    intensity: np.ndarray
    counts: np.ndarray
    candidate_counts: np.ndarray
    coverage: np.ndarray
    q_min: float
    q_max: float
    flags: tuple[str, ...] = APPARENT_FLAGS
    q_unit: str = "unknown"

    @property
    def profile(self) -> np.ndarray:
        return self.intensity


class RidgePoint(_MappingResult, _SharedRidgePoint):
    """A radial peak/ridge measured from one real angular sector.

    No counterpart mirrored from another quadrant is inserted.  ``source``
    records the number of pixels used for this point, and ``valid`` is false
    when the sector did not contain enough finite, unmasked signal.  The
    ``q_unit`` is deliberately fail-closed: direct callers must state a
    physical reciprocal-length unit before spacing fields are derived.
    Measured points receive the declaration propagated from their q map.
    Points from ``azimuthal_peak`` carry an annulus coordinate in ``q``; their
    ``q_star`` and lamellar-spacing aliases remain NaN because an angular
    maximum is not a measured radial reflection position.
    """

    def __init__(
        self,
        angle: float,
        q: float,
        q_star: float | None = None,
        lamellar_spacing: float | None = None,
        intensity: float = float("nan"),
        baseline: float = float("nan"),
        snr: float = float("nan"),
        radial_fwhm: float = float("nan"),
        azimuthal_fwhm: float = float("nan"),
        area: float = float("nan"),
        coverage: float = 0.0,
        n_pixels: int = 0,
        valid: bool = True,
        source: str = "observed",
        flags: tuple[str, ...] = APPARENT_FLAGS,
        method: str = "radial_peak",
        curvature: float = float("nan"),
        normal_slope: float = float("nan"),
        support: float = float("nan"),
        pixel_y: float = float("nan"),
        pixel_x: float = float("nan"),
        accepted: bool | None = None,
        reason: str = "",
        q_unit: str = "unknown",
        score: float = float("nan"),
        continuity_score: float = float("nan"),
        trajectory_id: int | None = None,
        branch_id: int | None = None,
        local_q_step: float = float("nan"),
        q_normal_step: float = float("nan"),
        q_scale_anisotropy: float = float("nan"),
    ) -> None:
        q = float(q)
        angle = float(angle)
        q_unit = str(q_unit or "unknown")
        q_for_coordinates = q if np.isfinite(q) else 0.0
        super().__init__(
            qx=q_for_coordinates * np.cos(angle),
            qy=q_for_coordinates * np.sin(angle),
            intensity=float(intensity) if np.isfinite(intensity) else None,
            weight=1.0,
            metadata={
                "angle": angle,
                "source": source,
                "flags": tuple(flags),
                "method": str(method),
                "q_unit": q_unit,
            },
        )
        self.angle = angle
        self.q = q
        self.q_star = q if q_star is None else float(q_star)
        q_scale = _q_to_nm_inverse_scale(q_unit)
        if lamellar_spacing is None:
            self.lamellar_spacing = (
                float(2.0 * np.pi / (q * q_scale))
                if q_scale is not None and q > 0
                else float("nan")
            )
        elif q_scale is None:
            # Do not retain an apparently physical spacing from an uncalibrated
            # q value merely because a caller supplied a legacy positional
            # value.
            self.lamellar_spacing = float("nan")
        else:
            self.lamellar_spacing = float(lamellar_spacing)
        self.baseline = float(baseline)
        self.snr = float(snr)
        self.radial_fwhm = float(radial_fwhm)
        self.azimuthal_fwhm = float(azimuthal_fwhm)
        self.area = float(area)
        self.coverage = float(coverage)
        self.n_pixels = int(n_pixels)
        self.valid = bool(valid)
        self.source = str(source)
        self.flags = tuple(flags)
        self.method = str(method)
        self.curvature = float(curvature)
        self.normal_slope = float(normal_slope)
        self.support = float(support)
        self.pixel_y = float(pixel_y)
        self.pixel_x = float(pixel_x)
        self.accepted = bool(valid if accepted is None else accepted)
        self.reason = str(reason or ("accepted" if self.accepted else "rejected"))
        self.q_unit = q_unit
        self.score = float(score)
        self.continuity_score = float(continuity_score)
        self.trajectory_id = None if trajectory_id is None else int(trajectory_id)
        self.branch_id = None if branch_id is None else int(branch_id)
        self.local_q_step = float(local_q_step)
        self.q_normal_step = float(q_normal_step)
        self.q_scale_anisotropy = float(q_scale_anisotropy)
        self.flags = _flags_with_q_unit(self.flags, q_unit)
        self.metadata.update(
            flags=self.flags,
            curvature=self.curvature,
            normal_slope=self.normal_slope,
            support=self.support,
            pixel_y=self.pixel_y,
            pixel_x=self.pixel_x,
            accepted=self.accepted,
            reason=self.reason,
            q_unit=self.q_unit,
            score=self.score,
            continuity_score=self.continuity_score,
            trajectory_id=self.trajectory_id,
            branch_id=self.branch_id,
            local_q_step=self.local_q_step,
            q_normal_step=self.q_normal_step,
            q_scale_anisotropy=self.q_scale_anisotropy,
        )

    def keys(self) -> tuple[str, ...]:
        return (
            "angle", "q", "q_star", "lamellar_spacing", "intensity", "baseline", "snr",
            "radial_fwhm", "azimuthal_fwhm", "area", "coverage", "n_pixels", "valid", "source", "flags",
            "qx", "qy", "method", "curvature", "normal_slope", "support",
            "pixel_y", "pixel_x", "accepted", "reason", "q_unit", "q_star_Ainv", "q_star_nm_inv", "Ln", "Ln_nm",
            "score", "point_score", "continuity_score", "trajectory_id", "branch_id", "local_q_step",
            "q_normal_step", "q_scale_anisotropy", "metadata",
        )

    @property
    def point_score(self) -> float:
        """Explicit alias for the point-level quality score."""

        return self.score

    @property
    def q_star_Ainv(self) -> float:
        """Peak position in inverse angstrom, when q has a physical unit."""

        scale = _q_to_nm_inverse_scale(self.q_unit)
        if scale is None or not np.isfinite(self.q_star):
            return float("nan")
        return float(self.q_star * scale / 10.0)

    @property
    def q_star_nm_inv(self) -> float:
        scale = _q_to_nm_inverse_scale(self.q_unit)
        if scale is None or not np.isfinite(self.q_star):
            return float("nan")
        return float(self.q_star * scale)

    @property
    def Ln(self) -> float:
        """Compatibility alias for :attr:`Ln_nm`."""

        return self.Ln_nm

    @property
    def Ln_nm(self) -> float:
        q_nm_inv = self.q_star_nm_inv
        if not np.isfinite(q_nm_inv) or q_nm_inv <= 0:
            return float("nan")
        return float(2.0 * np.pi / q_nm_inv)

    @property
    def q_position(self) -> float:
        return self.q


@dataclass
class RidgeTrack(_MappingResult):
    """Collection of radial ridge points and quality metadata."""

    points: list[RidgePoint]
    angles: np.ndarray
    q: np.ndarray
    valid: np.ndarray
    coverage: np.ndarray
    flags: tuple[str, ...] = APPARENT_FLAGS
    q_unit: str = "unknown"
    valid_fraction: float = float("nan")
    continuity_fraction: float = float("nan")
    continuity_score: float = float("nan")

    @property
    def observed_points(self) -> list[RidgePoint]:
        return [point for point in self.points if point.valid]


@dataclass
class EllipseGeometry(_MappingResult):
    """One apparent ellipse in q-space.

    ``a``, ``b`` and ``theta`` retain the original observables API.  The
    canonical solver also estimates a shared centre, so the adapter carries
    it without changing the positional constructor used by older notebooks.
    ``theta`` is always radians; :attr:`theta_deg` is the display value.
    """

    a: float
    b: float
    theta: float
    center: tuple[float, float] = (0.0, 0.0)

    @property
    def axes_ratio(self) -> float:
        return float(self.b / self.a) if self.a else float("nan")

    @property
    def ellipticity(self) -> float:
        """Grubb's eccentricity-valued ellipticity."""

        ratio = self.axes_ratio
        return float(np.sqrt(max(0.0, 1.0 - ratio * ratio))) if np.isfinite(ratio) else float("nan")

    @property
    def eccentricity(self) -> float:
        return self.ellipticity

    @property
    def theta_deg(self) -> float:
        return float(np.degrees(self.theta))

    @property
    def cx(self) -> float:
        return float(self.center[0])

    @property
    def cy(self) -> float:
        return float(self.center[1])


@dataclass
class DoubleEllipseFit(_MappingResult):
    """Symmetric pair of apparent ellipses fitted to ridge points."""

    a: float
    b: float
    theta: float
    ellipses: tuple[EllipseGeometry, EllipseGeometry]
    ellipticity: float
    axes_ratio: float
    rmse: float
    rss: float
    n_points: int
    branch_counts: tuple[int, int]
    success: bool
    message: str
    flags: tuple[str, ...] = APPARENT_FLAGS
    center: tuple[float, float] = (float("nan"), float("nan"))
    stderr: dict[str, float] = field(default_factory=dict)
    covariance: np.ndarray | None = None
    condition_number: float = float("nan")
    coverage: Any = None
    bound_flags: dict[str, bool] = field(default_factory=dict)
    bound_status: dict[str, str | None] = field(default_factory=dict)
    free_names: tuple[str, ...] = ()
    parameter_values: dict[str, float] = field(default_factory=dict)
    reference_axis_deg: float = 0.0
    q_unit: str = "unknown"
    Ln_from_minor_axis_nm: float = float("nan")
    Lz_from_draw_axis_nm: float = float("nan")
    branch_assignment: np.ndarray | None = None
    candidate_solutions: tuple[dict[str, Any], ...] = ()
    selected_start_index: int = 0
    multistart_count: int = 1
    quality: dict[str, Any] = field(default_factory=dict)
    symmetry: dict[str, Any] = field(default_factory=dict)
    branch_assignment_indices: np.ndarray | None = None

    @property
    def theta_deg(self) -> float:
        return float(np.degrees(self.theta))

    @property
    def ellipse_axis_tilt_deg(self) -> float:
        """Apparent ellipse-axis tilt relative to ``reference_axis_deg``."""

        return self.theta_deg

    @property
    def eccentricity(self) -> float:
        """Alias for the paper's eccentricity-valued ``ellipticity``."""

        return self.ellipticity

    @property
    def coverage_fraction(self) -> float:
        value = _get_field(self.coverage, ("angular_coverage", "coverage_fraction"), float("nan"))
        return float(value)

    @property
    def phi_app_deg(self) -> float:
        """Placeholder for an independently measured lamellar-tilt angle.

        The ellipse-axis tilt is not the same quantity as the apparent lobe
        angle (and neither is a direct measurement of the structural
        ``alpha``).  Returning NaN makes that distinction explicit for UI and
        export code until a lobe-aware inference step supplies it.
        """

        return float("nan")

    @property
    def alpha_candidate_deg(self) -> float:
        """No structural alpha is inferred by an apparent ellipse fit."""

        return float("nan")

    @property
    def ellipse_plus(self) -> EllipseGeometry:
        return self.ellipses[0]

    @property
    def ellipse_minus(self) -> EllipseGeometry:
        return self.ellipses[1]


@dataclass
class ObservableSet(_MappingResult):
    """High-level bundle used by a GUI or batch pipeline."""

    angular: AngularSpectrum
    lobes: list[LobeMetrics]
    ridge: RidgeTrack
    ellipse: DoubleEllipseFit | None
    phi_app_deg: float = float("nan")
    phi_app_std_deg: float = float("nan")
    draw_axis_deg: float = 90.0
    alpha_candidate_deg: float = float("nan")
    psi_candidate_deg: float = float("nan")
    flags: tuple[str, ...] = APPARENT_FLAGS
    q_unit: str = "unknown"
    # ``phi_app_std_deg`` is retained for compatibility; the estimator is a
    # scaled median absolute deviation, so expose the precise name as well.
    # Keep this new field at the end so legacy positional construction retains
    # its original argument order.
    phi_app_mad_deg: float = float("nan")
    # Azimuthal-peak tracks intentionally do not expose their annulus q as a
    # radial q_star.  These optional, directly observed lobe-sector profiles
    # retain the radial reflection position and width needed by callers that
    # want both measurements in one observable bundle.
    lobe_radial_profiles: list[RadialProfile] = field(default_factory=list)
    lobe_radial_peaks: list[RidgePoint] = field(default_factory=list)
    butterfly: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        std = float(self.phi_app_std_deg)
        mad = float(self.phi_app_mad_deg)
        if not np.isfinite(mad) and np.isfinite(std):
            mad = std
        elif not np.isfinite(std) and np.isfinite(mad):
            std = mad
        self.phi_app_std_deg = std
        self.phi_app_mad_deg = mad
        self.q_unit = str(self.q_unit or "unknown")


def _get_field(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """Read a field from either a mapping, an object, or a zero-arg property."""

    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            value = obj[name]
        elif hasattr(obj, name):
            value = getattr(obj, name)
        else:
            continue
        if callable(value):
            try:
                value = value()
            except TypeError:
                pass
        if value is not None:
            return value
    return default


def _q_unit(qmap: Any) -> str:
    unit = _get_field(qmap, ("q_unit", "unit"), None)
    if unit is None:
        metadata = _get_field(qmap, ("metadata",), {})
        unit = _get_field(metadata, ("q_unit", "unit"), "unknown")
    return str(unit or "unknown")




def _q_unit_spacing_flag(unit: Any) -> str | None:
    """Return a stable diagnostic when spacing cannot be physicalized."""

    if _q_to_nm_inverse_scale(unit) is not None:
        return None
    # Keep the public diagnostic stable for both uncalibrated pixel-q and any
    # other undeclared unit.  The q_unit field itself retains the more specific
    # provenance for callers that need to distinguish them.
    return "spacing_unavailable_unknown_q_unit"


def _flags_with_q_unit(flags: Sequence[str], unit: Any) -> tuple[str, ...]:
    """Attach the spacing diagnostic once to each q-bearing result layer."""

    result = tuple(flags)
    diagnostic = _q_unit_spacing_flag(unit)
    if diagnostic is not None and diagnostic not in result:
        result += (diagnostic,)
    if _normalized_q_unit(unit) in {"pixel-q", "pixel_q", "pixelq"} and "uncalibrated_pixel_q" not in result:
        result += ("uncalibrated_pixel_q",)
    return result


def _array_field(obj: Any, names: Sequence[str], default: Any = None) -> np.ndarray | None:
    value = _get_field(obj, names, default)
    if value is None:
        return None
    try:
        return np.asarray(value)
    except Exception:
        return None


def _extract_maps(frame: Any, qmap: Any, extra_mask: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened intensity, q, angle and validity arrays.

    ``mask=True`` follows detector conventions and means invalid.  A field
    named ``valid_mask`` follows its literal meaning and means valid.  This
    distinction avoids silently inverting the common pyFAI mask while still
    allowing simple test doubles to use a valid mask.
    """

    data = _array_field(frame, ("data", "intensity", "image", "values"))
    if data is None:
        raise ValueError("ImageFrame must expose data/intensity/image/values")
    data = np.asarray(data, dtype=float)
    shape = data.shape

    qx = _array_field(qmap, ("qx", "q_x", "x", "qx_map"))
    qy = _array_field(qmap, ("qy", "q_y", "y", "qy_map"))
    q = _array_field(qmap, ("q", "q_map", "q_abs", "radius"))
    angle = _array_field(qmap, ("angle", "azimuth", "chi", "phi", "azimuth_map"))
    if q is None and qx is not None and qy is not None:
        q = np.hypot(qx, qy)
    if angle is None and qx is not None and qy is not None:
        angle = np.arctan2(qy, qx)
    if q is None:
        raise ValueError("QMap must expose q or qx/qy arrays")
    if angle is None:
        raise ValueError("QMap must expose azimuth/angle or qx/qy arrays")

    q = np.broadcast_to(np.asarray(q, dtype=float), shape)
    angle = np.broadcast_to(np.asarray(angle, dtype=float), shape)
    data = np.broadcast_to(data, shape)
    valid = np.isfinite(data) & np.isfinite(q) & np.isfinite(angle)

    # A detector mask is normally True for bad pixels.
    detector_mask = _array_field(frame, ("mask", "bad_mask", "invalid_mask"))
    if detector_mask is not None:
        valid &= ~np.broadcast_to(np.asarray(detector_mask, dtype=bool), shape)
    qmask = _array_field(qmap, ("mask", "bad_mask", "invalid_mask"))
    if qmask is not None:
        valid &= ~np.broadcast_to(np.asarray(qmask, dtype=bool), shape)
    valid_mask = _array_field(frame, ("valid_mask", "valid"))
    if valid_mask is not None:
        valid_arr = np.asarray(valid_mask)
        if valid_arr.ndim > 0:
            if valid_arr.shape != shape:
                raise ValueError(
                    f"frame valid_mask shape {valid_arr.shape!r} does not match image shape {shape!r}"
                )
            valid &= np.asarray(valid_arr, dtype=bool)
    q_valid_mask = _array_field(qmap, ("valid_mask", "valid"))
    if q_valid_mask is not None:
        q_valid_arr = np.asarray(q_valid_mask)
        if q_valid_arr.ndim > 0:
            if q_valid_arr.shape != shape:
                raise ValueError(
                    f"qmap valid_mask shape {q_valid_arr.shape!r} does not match image shape {shape!r}"
                )
            valid &= np.asarray(q_valid_arr, dtype=bool)
    if extra_mask is not None:
        valid &= ~np.broadcast_to(np.asarray(extra_mask, dtype=bool), shape)

    return data.ravel(), q.ravel(), np.mod(angle.ravel() + np.pi, 2.0 * np.pi) - np.pi, valid.ravel()


def _q_limits(q: np.ndarray, q_window: Any = None, q_range: Any = None) -> tuple[float, float]:
    window = q_window if q_window is not None else q_range
    if window is None:
        finite = q[np.isfinite(q)]
        if not finite.size:
            raise ValueError("q map contains no finite values")
        return float(np.min(finite)), float(np.max(finite))
    if isinstance(window, Mapping):
        lo = _get_field(window, ("min", "q_min", "low", "start"))
        hi = _get_field(window, ("max", "q_max", "high", "stop"))
        if lo is None or hi is None:
            raise ValueError("q window mapping must provide min/max")
    else:
        try:
            lo, hi = window
        except Exception as exc:
            raise ValueError("q window must be a (min, max) pair") from exc
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("q window must contain finite max > min")
    return lo, hi


def _bin_mean(values: np.ndarray, indices: np.ndarray, n_bins: int, statistic: str) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values) & (indices >= 0) & (indices < n_bins)
    indices = indices[finite]
    values = values[finite]
    counts = np.bincount(indices, minlength=n_bins).astype(int)
    if statistic == "sum":
        profile = np.bincount(indices, weights=values, minlength=n_bins).astype(float)
        return profile, counts
    sums = np.bincount(indices, weights=values, minlength=n_bins).astype(float)
    profile = np.full(n_bins, np.nan, dtype=float)
    np.divide(sums, counts, out=profile, where=counts > 0)
    return profile, counts


def ellipse_radius(angle: np.ndarray | float, a: float, b: float, theta: float) -> np.ndarray:
    """Polar radius of an origin-centred, rotated ellipse."""

    angle = np.asarray(angle, dtype=float)
    a, b = float(a), float(b)
    if a <= 0 or b <= 0:
        return np.full_like(angle, np.nan, dtype=float)
    delta = angle - float(theta)
    denominator = np.sqrt((b * np.cos(delta)) ** 2 + (a * np.sin(delta)) ** 2)
    return a * b / denominator


__all__ = [
    "APPARENT_FLAGS",
    "_SPACING_CENTER_REL_TOL",
    "_MappingResult",
    "AngularSpectrum",
    "LobeMetrics",
    "RadialProfile",
    "RidgePoint",
    "RidgeTrack",
    "EllipseGeometry",
    "DoubleEllipseFit",
    "ObservableSet",
    "ellipse_radius",
    "_array_field",
    "_bin_mean",
    "_extract_maps",
    "_flags_with_q_unit",
    "_get_field",
    "_normalized_q_unit",
    "_q_limits",
    "_q_to_nm_inverse_scale",
    "_q_unit",
    "_q_unit_spacing_flag",
]
