"""
SIGAK 3-Axis Aesthetic Coordinate System

Axes:
  1. Shape  — Soft(-1) <-> Sharp(+1)   골격과 이목구비의 형태
  2. Volume — Subtle(-1) <-> Bold(+1)   이목구비의 크기와 볼륨
  3. Age    — Fresh(-1) <-> Mature(+1)  비율이 주는 나이 인상

All axis definitions loaded from sigak/data/axis_config.yaml (SSOT).
All observed ranges loaded from sigak/data/calibration_3axis.yaml, keyed by gender.

Gender-aware 리팩터 (2026-04-21):
  calibration_3axis.yaml 이 observed_ranges.{female,male} 중첩 구조.
  compute_coordinates(features, gender) 가 gender 별로 분기.
  해당 gender section 누락/미캘리브레이션 시 CoordinateConfigError raise.
"""
import math
import os
import yaml
from functools import lru_cache
from typing import Optional


class CoordinateConfigError(RuntimeError):
    """calibration_3axis.yaml 의 gender 섹션이 없거나 feature range 가 null 일 때.

    silent {0,0,0} 반환하지 않고 조기 실패시켜 왜곡된 좌표 산출을 차단.
    """


# ─────────────────────────────────────────────
#  YAML Config Loading (cached)
# ─────────────────────────────────────────────

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@lru_cache(maxsize=1)
def _load_axis_config() -> dict:
    """Load axis definitions from axis_config.yaml. Cached after first load."""
    path = os.path.join(_DATA_DIR, "axis_config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_calibration() -> dict:
    """Load calibration data from calibration_3axis.yaml. Cached after first load."""
    path = os.path.join(_DATA_DIR, "calibration_3axis.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
#  Axis Label Access (Single Source of Truth)
#  All other files MUST use these functions.
# ─────────────────────────────────────────────

def get_axis_labels(axis_name: str) -> dict:
    """축 라벨 조회. 모든 파일의 유일한 라벨 소스."""
    config = _load_axis_config()
    ax = config.get("axes", {}).get(axis_name)
    if ax is None:
        return {"name_kr": axis_name, "low": "", "high": "", "low_en": "", "high_en": "", "description": ""}
    return {
        "name_kr": ax["name_kr"],
        "low": ax["low_kr"],
        "high": ax["high_kr"],
        "low_en": ax["low"],
        "high_en": ax["high"],
        "description": ax.get("description_kr", ""),
    }


def get_all_axis_labels() -> dict:
    """전체 축 라벨. 프론트에 내려줄 때 사용."""
    config = _load_axis_config()
    return {name: get_axis_labels(name) for name in config["axes"]}


def get_axis_names() -> list[str]:
    """현재 축 이름 목록 반환. ["shape", "volume", "age"]"""
    config = _load_axis_config()
    return list(config["axes"].keys())


# ─────────────────────────────────────────────
#  Common Helpers (kept from original)
# ─────────────────────────────────────────────

def _has_valid(features: dict, key: str) -> bool:
    """feature가 존재하고 None이 아닌지 확인"""
    return key in features and features[key] is not None


def _normalize(value: float, observed_range) -> float:
    """관측 범위 기반 정규화. 범위 밖 값은 clamp. -> [-1, 1]

    observed_range can be a tuple (lo, hi) or a list [lo, hi].
    """
    if isinstance(observed_range, (list, tuple)) and len(observed_range) == 2:
        lo, hi = observed_range
    else:
        return 0.0
    if hi <= lo:
        return 0.0
    value = min(max(value, lo), hi)
    return (value - lo) / (hi - lo) * 2 - 1


def _weighted_fallback(components: list[tuple[float, float]]) -> float:
    """
    사용 가능한 component만으로 가중 평균.
    feature 미존재 시 나머지로 가중치 자동 재분배.
    """
    if not components:
        return 0.0
    weight_sum = sum(w for _, w in components)
    score = sum(v * w for v, w in components) / weight_sum
    return max(-1.0, min(1.0, score))


# ─────────────────────────────────────────────
#  Coordinate Computation (YAML-driven)
# ─────────────────────────────────────────────

def _resolve_obs_ranges(gender: str) -> dict:
    """calibration_3axis.yaml 의 observed_ranges.{gender} 를 로드 + 검증.

    - gender 섹션 자체가 없으면 CoordinateConfigError
    - 섹션은 있는데 axis_config 가 참조하는 feature 중 null 인 게 있으면
      CoordinateConfigError (미캘리브레이션 상태)

    통과 시 feature_name -> [lo, hi] dict 반환.
    """
    axis_config = _load_axis_config()
    cal = _load_calibration()
    all_ranges = cal.get("observed_ranges", {})

    if gender not in all_ranges:
        raise CoordinateConfigError(
            f"observed_ranges.{gender} not found in calibration_3axis.yaml. "
            f"Available: {list(all_ranges.keys())}"
        )
    obs_ranges = all_ranges[gender] or {}

    required = set()
    for axis_def in axis_config.get("axes", {}).values():
        for feat_name in axis_def.get("features", {}):
            required.add(feat_name)

    missing = []
    for feat_name in required:
        rng = obs_ranges.get(feat_name)
        if rng is None:
            missing.append(feat_name)
            continue
        if isinstance(rng, (list, tuple)) and (
            len(rng) != 2 or rng[0] is None or rng[1] is None
        ):
            missing.append(feat_name)

    if missing:
        subset_hint = {"female": "AF", "male": "AM"}.get(gender, gender.upper())
        raise CoordinateConfigError(
            f"observed_ranges.{gender} is not calibrated. "
            f"Missing/null features: {sorted(missing)}. "
            f"Run: python -m scripts.calibrate_face_stats --subset {subset_hint} "
            f"--image-dir <SCUT path> --output <json path>"
        )

    return obs_ranges


def compute_coordinates(
    structural_features: dict,
    gender: str = "female",
    clip_embedding=None,   # reserved for future CLIP integration
    projector=None,        # reserved for future CLIP integration
) -> dict[str, float]:
    """
    12개 피처 -> 3축 좌표 계산. gender 별로 다른 observed_ranges 적용.

    Args:
        structural_features: analyze_face().to_dict() 산출물
        gender: "female" / "male" — calibration 섹션 선택.
            미캘리브레이션된 gender 호출 시 CoordinateConfigError raise.
        clip_embedding, projector: 향후 CLIP projector 통합 예약 파라미터.

    Returns: {"shape": 0.35, "volume": -0.22, "age": 0.1}
    """
    axis_config = _load_axis_config()
    obs_ranges = _resolve_obs_ranges(gender)

    coords = {}
    for axis_name, axis_def in axis_config["axes"].items():
        components = []
        for feat_name, feat_def in axis_def["features"].items():
            raw = structural_features.get(feat_name)
            if raw is None:
                continue

            feat_range = obs_ranges.get(feat_name)
            if feat_range is None:
                # _resolve_obs_ranges 에서 이미 검증됐으므로 여기 도달 안 함.
                # 방어적으로 skip.
                continue

            normalized = _normalize(raw, feat_range)

            if feat_def["direction"] == "low_is_positive":
                normalized = -normalized

            components.append((normalized, feat_def["weight"]))

        coords[axis_name] = round(_weighted_fallback(components), 3)

    return coords


# ─────────────────────────────────────────────
#  Gap Computation
# ─────────────────────────────────────────────

def compute_gap(
    current_coords: dict[str, float],
    aspiration_coords: dict[str, float],
) -> dict:
    """
    현재 좌표와 추구미 좌표 간 gap 계산.

    Returns:
        {
            "vector": {"shape": 0.5, "volume": -0.3, "age": 0.8},
            "magnitude": 0.99,
            "primary_direction": "age",
            "primary_shift_kr": "성숙한",
            "secondary_direction": "shape",
            "secondary_shift_kr": "또렷한",
        }
    """
    axis_names = get_axis_names()

    # gap vector
    vector = {}
    for axis in axis_names:
        cur = current_coords.get(axis, 0.0)
        asp = aspiration_coords.get(axis, 0.0)
        vector[axis] = round(asp - cur, 3)

    # magnitude
    magnitude = math.sqrt(sum(v ** 2 for v in vector.values()))

    # sort by |delta| descending
    sorted_axes = sorted(vector.items(), key=lambda x: abs(x[1]), reverse=True)

    # primary direction
    primary_axis = sorted_axes[0][0]
    primary_delta = sorted_axes[0][1]
    primary_labels = get_axis_labels(primary_axis)
    primary_shift_kr = primary_labels["high"] if primary_delta > 0 else primary_labels["low"]

    # secondary direction
    secondary_axis = sorted_axes[1][0] if len(sorted_axes) > 1 else None
    secondary_shift_kr = ""
    if secondary_axis:
        secondary_delta = sorted_axes[1][1]
        secondary_labels = get_axis_labels(secondary_axis)
        secondary_shift_kr = secondary_labels["high"] if secondary_delta > 0 else secondary_labels["low"]

    return {
        "vector": vector,
        "magnitude": round(magnitude, 2),
        "primary_direction": primary_axis,
        "primary_shift_kr": primary_shift_kr,
        "secondary_direction": secondary_axis,
        "secondary_shift_kr": secondary_shift_kr,
    }
