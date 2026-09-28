"""PI 전용 methodology 어댑터 — Phase I PI-C.

CLAUDE.md §4.5 / §5.1 정의.

`pipeline/hair_rules.py` + `pipeline/action_spec.py` + `pipeline/personal_color.py`
+ `data/cluster_labels.json` 자산을 PI v1 어댑터들이 소비하기 좋은 순수 dataclass
형태로 통합. **pipeline 모듈은 read-only import 만 사용 — 절대 수정 X.**

순수 함수. side effect / LLM 호출 / R2 호출 없음. Day 1 fallback 안전 보장.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pipeline.action_spec import (
    AXIS_ACTION_RULES_FEMALE,
    AXIS_ACTION_RULES_MALE,
)
from pipeline.hair_rules import FEATURE_MODIFIERS
from pipeline.hair_styles import HAIR_STYLES
from pipeline.personal_color import (
    PersonalColorClassifier,
    PersonalColorResult,
    SEASON_PALETTES,
    get_season_palette,
)
from schemas.knowledge import Gender
from services.coordinate_system import VisualCoordinate


# 파일 경로 — pipeline/* 와 동일 베이스
_DATA_ROOT = Path(__file__).parent.parent / "data"
_CLUSTER_LABELS_PATH = _DATA_ROOT / "cluster_labels.json"


# ─────────────────────────────────────────────
#  Result containers
# ─────────────────────────────────────────────

@dataclass
class HairMethodologyEntry:
    """단일 헤어 추천 — hair_rules.FEATURE_MODIFIERS 누적 점수 결과."""
    hair_id: str
    hair_name: str
    score: float
    reason: str                           # 매칭된 modifier 의 reason 결합
    matched_features: list[str]           # 활성 feature 리스트


@dataclass
class ActionMethodologyEntry:
    """단일 action — AXIS_ACTION_RULES 매칭 결과."""
    title: str
    description: str
    zone: str
    method: str
    goal: str
    source: str                           # axis 또는 zone 기반 출처


@dataclass
class ColorMethodologyEntry:
    """퍼스널 컬러 분류 결과 + 팔레트."""
    best: list[str]                       # hex 5
    ok: list[str]                         # hex 5
    avoid: list[str]                      # hex 5
    foundation: str
    lip_cheek_eye: dict[str, str]         # {lip, cheek, eye}
    classification_reason: str            # PersonalColorClassifier 산출 라벨


@dataclass
class MethodologyResult:
    """`derive_methodology` 결과 — PI 어댑터들이 소비하는 컨테이너."""
    hair: list[HairMethodologyEntry] = field(default_factory=list)
    action: list[ActionMethodologyEntry] = field(default_factory=list)
    cluster_keywords: list[str] = field(default_factory=list)
    color: ColorMethodologyEntry = field(
        default_factory=lambda: _neutral_color_entry()
    )


# ─────────────────────────────────────────────
#  Public entry
# ─────────────────────────────────────────────

def derive_methodology(
    face_features: dict,
    coord: VisualCoordinate,
    gender: Gender,
    *,
    hair_limit: int = 5,
    action_limit: int = 5,
) -> MethodologyResult:
    """face_features + 좌표 + 성별 → 헤어/액션/클러스터/컬러 통합.

    - face_features: PI-B FaceMetric.dict() 또는 그 호환 dict.
      비어있거나 키 누락 시 안전 fallback (raise 하지 않음).
    - coord: 3축 좌표 (외부 0~1 스케일).
    - gender: female | male — AXIS_ACTION_RULES_* 분기.

    pipeline/* 의 entry 함수 호출 시 외부 자산 (hair_styles_json / type_anchors)
    이 필요해 모듈 직접 사용. PI-C 는 raw 자료구조만 직접 매칭하여 부수 효과 차단.
    """
    safe_features = face_features if isinstance(face_features, dict) else {}
    is_empty = not safe_features

    hair_entries = _derive_hair(safe_features, gender, limit=hair_limit)
    action_entries = _derive_actions(coord, gender, limit=action_limit)
    cluster_keywords = _derive_cluster_keywords(coord)
    color_entry = _derive_color(safe_features) if not is_empty else _neutral_color_entry()

    return MethodologyResult(
        hair=hair_entries,
        action=action_entries,
        cluster_keywords=cluster_keywords,
        color=color_entry,
    )


# ─────────────────────────────────────────────
#  Hair — FEATURE_MODIFIERS 누적 + 컬렉션
# ─────────────────────────────────────────────

# face_features 키 → FEATURE_MODIFIERS 키 (pipeline/hair_spec._categorize_face_features
# 와 의도 동일하지만 본 모듈은 독립). raise 회피를 위해 단순 매핑.
_FACE_FEATURE_TO_MODIFIER = {
    # 얼굴 비율
    "wide_face": "face_wide_short",
    "short_face": "face_wide_short",
    "face_wide_short": "face_wide_short",
    # 이마
    "short_forehead": "short_forehead",
    # 중안부 / 인중
    "long_midface": "long_midface",
    "long_philtrum": "long_philtrum",
    # 턱
    "square_jaw": "square_jaw",
    # 입돌출
    "mouth_protrusion": "mouth_protrusion",
    # 코
    "large_nose": "large_nose",
    # 목
    "short_neck": "short_neck",
    # 어깨
    "narrow_shoulders": "narrow_shoulders",
}


def _activate_features(face_features: dict) -> list[str]:
    """face_features dict 에서 활성화된 FEATURE_MODIFIERS 키만 추출."""
    active: list[str] = []

    # bool 또는 truthy 값으로 표시된 키
    for src_key, mod_key in _FACE_FEATURE_TO_MODIFIER.items():
        v = face_features.get(src_key)
        if v is True or (isinstance(v, str) and v.lower() in ("yes", "true", "y")):
            if mod_key not in active:
                active.append(mod_key)

    # 수치 기반 — face.py FaceFeatures 호환
    fr = face_features.get("forehead_ratio")
    if isinstance(fr, (int, float)) and fr < 0.28 and "short_forehead" not in active:
        active.append("short_forehead")

    philtrum = face_features.get("philtrum_ratio")
    if (
        isinstance(philtrum, (int, float))
        and philtrum > 0.38
        and "long_philtrum" not in active
    ):
        active.append("long_philtrum")

    jaw_angle = face_features.get("jaw_angle")
    if isinstance(jaw_angle, (int, float)) and jaw_angle < 115 and "square_jaw" not in active:
        active.append("square_jaw")

    nose_w = face_features.get("nose_width_ratio")
    if isinstance(nose_w, (int, float)) and nose_w > 0.30 and "large_nose" not in active:
        active.append("large_nose")

    flr = face_features.get("face_length_ratio")
    if (
        isinstance(flr, (int, float))
        and flr < 1.05
        and "face_wide_short" not in active
    ):
        active.append("face_wide_short")

    return active


def _derive_hair(
    face_features: dict,
    gender: Gender,
    *,
    limit: int = 5,
) -> list[HairMethodologyEntry]:
    """active features → FEATURE_MODIFIERS 누적 점수 → top hair_limit."""
    active_features = _activate_features(face_features)

    # 후보 풀 — gender 필터.
    pool: dict[str, dict] = {
        sid: meta
        for sid, meta in HAIR_STYLES.items()
        if meta.get("gender", "female") == gender
        and meta.get("has_rating") is not False
    }

    if not active_features or not pool:
        return []

    # 각 style 별 base 0.5 + Σ feature_modifiers
    scored: list[tuple[str, float, list[str], list[str]]] = []
    for style_id, meta in pool.items():
        base = float(meta.get("base_score", 0.5))
        score = base
        applied_reasons: list[str] = []
        applied_features: list[str] = []

        for feat in active_features:
            feat_table = FEATURE_MODIFIERS.get(feat, {})
            entry = feat_table.get(style_id)
            if not entry:
                continue
            mod_value = float(entry.get("mod", 0.0))
            score += mod_value
            reason = entry.get("reason", "")
            if reason and reason not in applied_reasons:
                applied_reasons.append(reason)
            if feat not in applied_features:
                applied_features.append(feat)

        # raw score 유지 (clamp 없음)
        if applied_features:
            scored.append(
                (style_id, round(score, 3), applied_reasons, applied_features)
            )

    scored.sort(key=lambda t: t[1], reverse=True)

    out: list[HairMethodologyEntry] = []
    for style_id, score, reasons, feats in scored[:limit]:
        meta = pool[style_id]
        out.append(
            HairMethodologyEntry(
                hair_id=style_id,
                hair_name=meta.get("name_kr", style_id),
                score=score,
                reason=". ".join(reasons[:3]) if reasons else "활성 특징 없음",
                matched_features=feats,
            )
        )
    return out


# ─────────────────────────────────────────────
#  Action — AXIS_ACTION_RULES 매칭
# ─────────────────────────────────────────────

def _derive_actions(
    coord: VisualCoordinate,
    gender: Gender,
    *,
    limit: int = 5,
) -> list[ActionMethodologyEntry]:
    """좌표가 0.5 (중간) 에서 벗어난 축마다 increase/decrease 룰을 매칭.

    좌표 0~1 외부 스케일에서 0.5 기준. 0.5 위 → increase, 아래 → decrease.
    |delta| 가 큰 축부터 우선. limit 까지 zone 중복 제거.
    """
    rules = AXIS_ACTION_RULES_MALE if gender == "male" else AXIS_ACTION_RULES_FEMALE

    deltas = [
        ("shape", coord.shape - 0.5),
        ("volume", coord.volume - 0.5),
        ("age", coord.age - 0.5),
    ]
    deltas.sort(key=lambda t: abs(t[1]), reverse=True)

    candidates: list[tuple[float, dict, str, str]] = []
    for axis, delta in deltas:
        if abs(delta) < 0.05:
            continue
        direction = "increase" if delta > 0 else "decrease"
        axis_rules = rules.get(axis, {}).get(direction, [])
        for rule in axis_rules:
            score = float(rule.get("base_score", 0.5)) * abs(delta)
            candidates.append((score, rule, axis, direction))

    candidates.sort(key=lambda t: t[0], reverse=True)

    out: list[ActionMethodologyEntry] = []
    seen_zones: set[str] = set()
    for score, rule, axis, direction in candidates:
        zone = rule.get("zone", "")
        if zone in seen_zones:
            continue
        seen_zones.add(zone)
        out.append(
            ActionMethodologyEntry(
                title=rule.get("goal", ""),
                description=f"{axis} 축 {direction} 방향. {rule.get('goal', '')}",
                zone=zone,
                method=rule.get("method", ""),
                goal=rule.get("goal", ""),
                source=f"axis:{axis}/{direction}",
            )
        )
        if len(out) >= limit:
            break
    return out


# ─────────────────────────────────────────────
#  Cluster keywords — data/cluster_labels.json
# ─────────────────────────────────────────────

_cluster_cache: Optional[dict[str, Any]] = None


def _load_cluster_labels() -> dict[str, Any]:
    global _cluster_cache
    if _cluster_cache is None:
        try:
            with _CLUSTER_LABELS_PATH.open("r", encoding="utf-8") as fh:
                _cluster_cache = json.load(fh)
        except (OSError, json.JSONDecodeError):
            _cluster_cache = {}
    return _cluster_cache


def _derive_cluster_keywords(coord: VisualCoordinate) -> list[str]:
    """3축 좌표 → 가장 가까운 cluster centroid → keywords.

    cluster_labels.json centroid_coords 는 내부 스케일 (-1~+1) 기준.
    유저 coord (0~1) 를 (-1~+1) 로 변환 후 Euclidean 최단 클러스터.
    """
    data = _load_cluster_labels()
    clusters = data.get("clusters", [])
    if not clusters:
        return ["뉴트럴"]

    # 외부 0~1 → 내부 -1~+1
    si = coord.shape * 2 - 1
    vi = coord.volume * 2 - 1
    ai = coord.age * 2 - 1

    best = None
    best_dist = float("inf")
    for cluster in clusters:
        centroid = cluster.get("centroid_coords") or {}
        cs = centroid.get("shape", 0.0)
        cv = centroid.get("volume", 0.0)
        ca = centroid.get("age", 0.0)
        dist = ((si - cs) ** 2 + (vi - cv) ** 2 + (ai - ca) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best = cluster

    if not best:
        return ["뉴트럴"]
    keywords = best.get("keywords") or [best.get("label_kr", "뉴트럴")]
    return list(keywords)


# ─────────────────────────────────────────────
#  Color — PersonalColorClassifier + SEASON_PALETTES
# ─────────────────────────────────────────────

def _neutral_color_entry() -> ColorMethodologyEntry:
    """face_features 부재 시 안전 fallback — 봄 라이트 팔레트 기본값."""
    palette = get_season_palette("spring", "light")
    return ColorMethodologyEntry(
        best=_extract_hex_list(palette.get("best_colors", []), max_n=5),
        ok=_extract_hex_list(palette.get("okay_colors", []), max_n=5),
        avoid=_extract_hex_list(palette.get("avoid_colors", []), max_n=5),
        foundation=str(palette.get("foundation_guide", "")),
        lip_cheek_eye=_lip_cheek_eye_dict(palette),
        classification_reason="insufficient data — neutral fallback",
    )


def _extract_hex_list(items: list[dict], *, max_n: int = 5) -> list[str]:
    out: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        hex_v = item.get("hex")
        if isinstance(hex_v, str) and hex_v.startswith("#") and hex_v not in out:
            out.append(hex_v)
        if len(out) >= max_n:
            break
    return out


def _lip_cheek_eye_dict(palette: dict) -> dict[str, str]:
    return {
        "lip": str(palette.get("lip_direction", "")),
        "cheek": str(palette.get("cheek_direction", "")),
        "eye": str(palette.get("eye_direction", "")),
    }


def _derive_color(face_features: dict) -> ColorMethodologyEntry:
    """face_features 의 skin_warmth_score / skin_brightness / skin_chroma →
    PersonalColorClassifier → SEASON_PALETTES 매칭."""
    warmth = face_features.get("skin_warmth_score")
    brightness = face_features.get("skin_brightness")
    chroma = face_features.get("skin_chroma")

    # 키 누락 시 neutral fallback
    if warmth is None or brightness is None or chroma is None:
        return _neutral_color_entry()

    # face.py 는 skin_brightness 를 0~1 로 저장 (l_mean / 100 추정).
    # PersonalColorClassifier.classify 는 0~100 스케일 L 을 받음 → 변환.
    try:
        brightness_pct = float(brightness)
        if 0.0 <= brightness_pct <= 1.0:
            brightness_pct *= 100.0
    except (TypeError, ValueError):
        return _neutral_color_entry()

    try:
        result: PersonalColorResult = PersonalColorClassifier().classify(
            warmth=float(warmth),
            brightness=brightness_pct,
            chroma=float(chroma),
            is_calibrated=False,
        )
    except (TypeError, ValueError):
        return _neutral_color_entry()

    palette = get_season_palette(result.season, result.subtype)
    classification_reason = (
        f"{result.label_kr} (warmth_z={result.warmth_z}, "
        f"brightness={result.brightness}, chroma={result.chroma}, "
        f"confidence={result.confidence})"
    )

    return ColorMethodologyEntry(
        best=_extract_hex_list(palette.get("best_colors", []), max_n=5),
        ok=_extract_hex_list(palette.get("okay_colors", []), max_n=5),
        avoid=_extract_hex_list(palette.get("avoid_colors", []), max_n=5),
        foundation=str(palette.get("foundation_guide", "")),
        lip_cheek_eye=_lip_cheek_eye_dict(palette),
        classification_reason=classification_reason,
    )


# ─────────────────────────────────────────────
#  v1.5 후속 — pipeline → KB yaml export stub
# ─────────────────────────────────────────────

def export_methodology_to_yaml(result: MethodologyResult, out_path: Path) -> None:
    """v1.5 후속 — pipeline 코드 자산 → KB yaml 변환. 현재는 NotImplementedError.

    인터페이스만 미리 잡아둠. 실 구현 시 hair_rules / action_spec 의 자료구조를
    services/knowledge_base/methodology/{gender}/*.yaml 형태로 직렬화.
    """
    raise NotImplementedError("v1.5 후속 작업 — KB yaml export pipeline → kb 변환")
