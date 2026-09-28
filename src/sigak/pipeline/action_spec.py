"""
SIGAK Action Spec Layer

의사결정 우선순위:
1. 유저 목표 방향은 gap이 결정
2. 어느 부위를 우선할지는 type_delta + type_anchors.json coords 기반 modifier가 보정
3. 실행 강도는 modifier.shading_intensity가 조정

함수 분리:
- build_action_spec() → 의미 레이어 (방향 + 부위 + 강도)
- build_overlay_plan() → 시각 레이어 (zone name + 색상 + 투명도)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────

@dataclass
class VisualPriority:
    zone: str
    importance: float
    reason: str
    axis: str
    delta_direction: str  # "increase" | "decrease"


@dataclass
class RecommendedAction:
    zone: str
    method: str
    goal: str
    priority: int  # 1, 2, 3, 4


@dataclass
class AvoidAction:
    zone: str
    method: str
    reason: str


@dataclass
class ActionSpec:
    matched_type_id: str
    matched_type_label: str
    primary_gap_axis: str
    visual_priorities: list[VisualPriority]       # 최대 3개
    recommended_actions: list[RecommendedAction]   # 2~4개 (최소 2개 보장)
    avoid_actions: list[AvoidAction]               # 최대 2개
    expected_effects: list[str]                    # 2~3문장
    report_mode: str                               # "makeup_female_v0"
    _debug_trace: dict = field(default_factory=dict, repr=False)


@dataclass
class OverlayZone:
    zone_name: str     # semantic name (landmark indices 없음)
    zone_type: str     # "shading" | "blush" | "highlight" | "tint"
    color_hex: str
    opacity: float


# ─────────────────────────────────────────────
#  축별 Action 룰 테이블 — Female (메이크업)
# ─────────────────────────────────────────────

AXIS_ACTION_RULES_FEMALE: dict[str, dict[str, list[dict]]] = {
    "shape": {
        "increase": [  # → Sharp
            {"zone": "jawline", "method": "contour_shading", "goal": "턱선 정리로 윤곽 선명화", "base_score": 0.9},
            {"zone": "nose_bridge", "method": "highlight", "goal": "코 중심축 강조", "base_score": 0.7},
            {"zone": "brow_tail", "method": "sharp_draw", "goal": "눈썹 끝 정리로 선명한 인상", "base_score": 0.6},
        ],
        "decrease": [  # → Soft
            {"zone": "cheek_apple", "method": "round_blush", "goal": "볼 중심 볼륨감으로 부드러운 인상", "base_score": 0.9},
            {"zone": "jawline", "method": "highlight", "goal": "턱선 경계 완화", "base_score": 0.7},
            {"zone": "brow_arch", "method": "rounded_draw", "goal": "눈썹 곡선화로 부드러운 인상", "base_score": 0.6},
        ],
    },
    "volume": {
        "increase": [  # → Bold
            {"zone": "eye_crease", "method": "shadow_depth", "goal": "눈두덩 음영으로 깊은 눈매", "base_score": 0.9},
            {"zone": "lip", "method": "full_color", "goal": "립 컬러 강조로 존재감", "base_score": 0.75},
            {"zone": "nose_tip", "method": "subtle_shadow", "goal": "코끝 음영으로 오목한 느낌", "base_score": 0.6},
        ],
        "decrease": [  # → Subtle
            {"zone": "overall", "method": "matte_base", "goal": "전체 매트 베이스로 차분한 느낌", "base_score": 0.9},
            {"zone": "lip", "method": "nude_tone", "goal": "자연스러운 립 톤으로 부담감 완화", "base_score": 0.75},
            {"zone": "brow", "method": "feathered_draw", "goal": "자연스러운 눈썹결로 힘 빼기", "base_score": 0.6},
        ],
    },
    "age": {
        "increase": [  # → Mature
            {"zone": "cheekbone", "method": "contour_shading", "goal": "광대 음영으로 성숙한 윤곽", "base_score": 0.85},
            {"zone": "brow", "method": "straight_draw", "goal": "일자 눈썹으로 정돈된 인상", "base_score": 0.7},
            {"zone": "lip", "method": "defined_line", "goal": "립라인 정리로 단정한 느낌", "base_score": 0.6},
        ],
        "decrease": [  # → Fresh
            {"zone": "cheek_apple", "method": "bright_blush", "goal": "볼 사과존 블러셔로 어려 보이는 효과", "base_score": 0.85},
            {"zone": "under_eye", "method": "bright_concealer", "goal": "눈 밑 밝기로 동안 느낌", "base_score": 0.7},
            {"zone": "lip_center", "method": "gloss_highlight", "goal": "입술 중앙 광택으로 생기", "base_score": 0.6},
        ],
    },
}


# ─────────────────────────────────────────────
#  축별 Action 룰 테이블 — Male (그루밍/헤어/스킨케어)
# ─────────────────────────────────────────────

AXIS_ACTION_RULES_MALE: dict[str, dict[str, list[dict]]] = {
    "shape": {
        "increase": [  # → Sharp
            {"zone": "hair", "method": "short_sides", "goal": "사이드를 짧게 정리해 얼굴 윤곽 선명화", "base_score": 0.9},
            {"zone": "brow", "method": "clean_trim", "goal": "눈썹 끝 라인 정리로 또렷한 인상", "base_score": 0.7},
            {"zone": "beard", "method": "jawline_accent", "goal": "턱선 따라 수염 정리로 각진 인상 강조", "base_score": 0.6},
            {"zone": "body", "method": "lean_diet", "goal": "체지방 감량으로 턱선과 얼굴 윤곽 선명화", "base_score": 0.5},
        ],
        "decrease": [  # → Soft
            {"zone": "hair", "method": "textured_fringe", "goal": "앞머리와 질감으로 이마 라인을 부드럽게", "base_score": 0.9},
            {"zone": "brow", "method": "natural_shape", "goal": "눈썹 자연스럽게 유지, 과도한 정리 지양", "base_score": 0.7},
            {"zone": "skin", "method": "moisturize", "goal": "충분한 보습으로 피부결 부드럽게", "base_score": 0.6},
        ],
    },
    "volume": {
        "increase": [  # → Bold
            {"zone": "brow", "method": "fill_shape", "goal": "눈썹 빈 곳 채워 또렷한 눈매 강조", "base_score": 0.9},
            {"zone": "hair", "method": "volume_styling", "goal": "볼륨감 있는 스타일링으로 존재감 강화", "base_score": 0.75},
            {"zone": "body", "method": "bulk_training", "goal": "상체 볼륨업으로 전체 존재감 강화 (어깨/가슴)", "base_score": 0.65},
            {"zone": "skin", "method": "tone_up", "goal": "톤업 선크림으로 피부 밝기 올려 이목구비 부각", "base_score": 0.5},
        ],
        "decrease": [  # → Subtle
            {"zone": "hair", "method": "neat_down", "goal": "차분하게 눌린 스타일로 절제된 느낌", "base_score": 0.9},
            {"zone": "brow", "method": "thin_natural", "goal": "눈썹 얇게 정리해 부담 없는 인상", "base_score": 0.75},
            {"zone": "body", "method": "lean_cardio", "goal": "유산소 위주로 슬림하고 절제된 체형 유지", "base_score": 0.55},
            {"zone": "skin", "method": "matte_care", "goal": "매트한 피부 관리로 차분한 느낌", "base_score": 0.5},
        ],
    },
    "age": {
        "increase": [  # → Mature
            {"zone": "hair", "method": "slick_back", "goal": "이마를 드러내는 스타일로 성숙한 인상", "base_score": 0.85},
            {"zone": "beard", "method": "stubble_maintain", "goal": "짧은 수염으로 성숙하고 남성적인 느낌", "base_score": 0.7},
            {"zone": "brow", "method": "straight_shape", "goal": "일자 눈썹으로 정돈된 인상", "base_score": 0.6},
            {"zone": "body", "method": "posture_shoulder", "goal": "어깨 펴고 자세 교정으로 성숙한 체형", "base_score": 0.5},
        ],
        "decrease": [  # → Fresh
            {"zone": "hair", "method": "soft_bangs", "goal": "부드러운 앞머리로 어려 보이는 효과", "base_score": 0.85},
            {"zone": "beard", "method": "clean_shave", "goal": "깔끔한 면도로 동안 느낌", "base_score": 0.7},
            {"zone": "skin", "method": "hydra_glow", "goal": "수분 광채 케어로 생기 있는 피부", "base_score": 0.6},
            {"zone": "body", "method": "lean_maintain", "goal": "슬림한 체형 유지로 가벼운 인상", "base_score": 0.5},
        ],
    },
}


# ─────────────────────────────────────────────
#  Gender별 룰 선택 헬퍼
# ─────────────────────────────────────────────

def _get_action_rules(gender: str) -> dict:
    """gender에 따라 적절한 AXIS_ACTION_RULES 반환."""
    if gender == "male":
        return AXIS_ACTION_RULES_MALE
    return AXIS_ACTION_RULES_FEMALE


MAX_TARGET_AXES = 3
MIN_RECOMMENDED_ACTIONS = 2
MAX_RECOMMENDED_ACTIONS = 4


# ─────────────────────────────────────────────
#  타입별 modifier (type_anchors.json coords에서 동적 도출)
# ─────────────────────────────────────────────

_TYPE_ANCHORS_PATH = Path(__file__).parent.parent / "data" / "type_anchors.json"


def _load_type_anchors() -> dict:
    with open(_TYPE_ANCHORS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _derive_type_modifier(coords: dict) -> dict:
    """앵커 좌표에서 스타일 modifier를 도출한다."""
    shape_val = coords.get("shape", 0)
    volume_val = coords.get("volume", 0)
    age_val = coords.get("age", 0)

    # shading intensity: shape 축 기반
    if shape_val < -0.3:
        shading = "minimal"
    elif shape_val < 0.3:
        shading = "light"
    else:
        shading = "medium"

    # zone boost: 좌표 크기에 따라 관련 zone 가중
    zone_boost: dict[str, float] = {}
    if volume_val > 0.3:
        zone_boost["eye_crease"] = 0.2
        zone_boost["lip"] = 0.15
    if volume_val < -0.3:
        zone_boost["overall"] = 0.15
    if shape_val > 0.3:
        zone_boost["jawline"] = 0.2
        zone_boost["nose_bridge"] = 0.1
    if shape_val < -0.3:
        zone_boost["cheek_apple"] = 0.2
    if age_val > 0.3:
        zone_boost["cheekbone"] = 0.15
    if age_val < -0.3:
        zone_boost["under_eye"] = 0.15

    # style tone: 상위 2개 톤 조합
    tones: list[str] = []
    tones.append("부드러운" if shape_val < 0 else "또렷한")
    tones.append("강렬한" if volume_val > 0 else "은은한")
    tones.append("성숙한" if age_val > 0 else "발랄한")
    style_tone = " · ".join(tones[:2])

    return {
        "shading_intensity": shading,
        "zone_boost": zone_boost,
        "style_tone": style_tone,
        "avoid_override": [],
    }


def get_type_modifier(type_key: str) -> dict:
    """특정 앵커 타입의 modifier를 coords에서 도출하여 반환한다."""
    data = _load_type_anchors()
    anchor = data.get("anchors", {}).get(type_key)
    if not anchor:
        return {"shading_intensity": "light", "zone_boost": {}, "style_tone": "", "avoid_override": []}
    return _derive_type_modifier(anchor["coords"])


# ─────────────────────────────────────────────
#  Overlay 시각 매핑
# ─────────────────────────────────────────────

ZONE_VISUAL: dict[str, tuple[str, str, float]] = {
    "jawline":         ("shading",   "#8B6914", 0.25),
    "temple":          ("shading",   "#8B6914", 0.20),
    "cheekbone":       ("shading",   "#8B6914", 0.22),
    "mid_cheek":       ("blush",     "#F4B8C1", 0.22),
    "outer_cheek":     ("blush",     "#F4B8C1", 0.22),
    "cheek_apple":     ("blush",     "#F4B8C1", 0.22),
    "outer_eye":       ("shading",   "#9B7653", 0.20),
    "under_eye":       ("highlight", "#F5E6D0", 0.25),
    "nose_bridge":     ("highlight", "#F5E6D0", 0.30),
    "forehead_center": ("highlight", "#F5E6D0", 0.25),
    "brow_tail":       ("shading",   "#7B6544", 0.18),
    "eye_crease":      ("shading",   "#8B7355", 0.22),
    "lip":             ("tint",      "#C4616C", 0.30),
    "lip_center":      ("highlight", "#F0D0C8", 0.25),
    "lip_corner":      ("shading",   "#A0705A", 0.15),
    "nose_tip":        ("shading",   "#9B8060", 0.18),
    "brow":            ("shading",   "#6B5A3A", 0.20),
    "brow_arch":       ("shading",   "#6B5A3A", 0.18),
    "overall":         ("shading",   "#C8B898", 0.10),
}


# ─────────────────────────────────────────────
#  Internal Helpers
# ─────────────────────────────────────────────

def _derive_zone_bonus(type_delta: dict[str, float]) -> dict[str, float]:
    """
    타입 대비 delta가 큰 축 → 관련 zone에 bonus.
    magnitude 기반 (v0). 방향 필터링은 다음 스프린트에서 정교화.
    """
    AXIS_ZONE_MAP = {
        "shape": ["jawline", "temple", "cheekbone", "nose_bridge", "brow_tail"],
        "volume": ["eye_crease", "lip", "nose_tip", "brow"],
        "age": ["cheekbone", "brow", "cheek_apple", "under_eye"],
    }
    zone_bonus: dict[str, float] = {}
    for axis, delta in type_delta.items():
        if abs(delta) < 0.1:
            continue
        bonus = abs(delta) * 0.3
        for zone in AXIS_ZONE_MAP.get(axis, []):
            zone_bonus[zone] = zone_bonus.get(zone, 0) + bonus
    return zone_bonus


def _generate_expected_effects(
    primary_axis: str, type_label: str, style_tone: str
) -> list[str]:
    """deterministic 기대 효과 문장 생성"""
    axis_effect = {
        "shape": "얼굴 윤곽이 더 정돈돼 보입니다",
        "volume": "이목구비의 존재감이 조정됩니다",
        "age": "전체적인 분위기가 달라져 보입니다",
    }
    effects = [axis_effect.get(primary_axis, "전체 인상이 개선됩니다")]
    if type_label and style_tone:
        effects.append(f"{type_label} 타입의 {style_tone} 매력이 더 살아납니다")
    effects.append("적용 순서대로 하나씩 시도해보세요")
    return effects


# ─────────────────────────────────────────────
#  build_action_spec()
# ─────────────────────────────────────────────

def build_action_spec(
    face_features,
    current_coords: dict[str, float],
    matched_type: dict,
    type_delta: dict[str, float],
    gap: dict,
    interview_intent: dict[str, str] | None = None,
    gender: str = "female",
) -> ActionSpec:
    """
    Layer 1: gap → 목표 방향 결정
    Layer 2: type_delta + modifier → 부위 우선순위 보정
    """
    debug_trace: dict = {}

    # ── Layer 1: gap에서 target axes 추출 ──
    gap_vector = gap.get("vector", {})
    sorted_axes = sorted(gap_vector.items(), key=lambda x: abs(x[1]), reverse=True)
    target_axes: list[tuple[str, str, float]] = []
    for axis, delta_val in sorted_axes:
        if abs(delta_val) < 0.1:
            continue
        direction = "increase" if delta_val > 0 else "decrease"
        target_axes.append((axis, direction, abs(delta_val)))
    target_axes = target_axes[:MAX_TARGET_AXES]

    primary_axis = target_axes[0][0] if target_axes else "shape"
    debug_trace["target_axes"] = [(a, d, round(m, 3)) for a, d, m in target_axes]

    # ── Layer 1.5: gender별 룰에서 action 후보 + base_score 수집 ──
    action_rules = _get_action_rules(gender)
    candidates: list[dict] = []
    for axis, direction, magnitude in target_axes:
        rules = action_rules.get(axis, {}).get(direction, [])
        for rule in rules:
            score = rule["base_score"] * magnitude
            candidates.append({
                **rule, "axis": axis, "direction": direction, "score": score,
            })

    # ── Layer 2: type_delta로 zone importance 보정 ──
    type_zone_bonus = _derive_zone_bonus(type_delta)
    for c in candidates:
        c["score"] += type_zone_bonus.get(c["zone"], 0.0)
    debug_trace["type_zone_bonus"] = type_zone_bonus

    # ── Layer 2.5: 타입 modifier 적용 ──
    type_id = matched_type.get("key", "")
    modifier = get_type_modifier(type_id)

    for c in candidates:
        c["score"] += modifier.get("zone_boost", {}).get(c["zone"], 0.0)

    avoid_actions: list[AvoidAction] = []
    for override in modifier.get("avoid_override", []):
        avoid_actions.append(AvoidAction(**override))
        candidates = [
            c for c in candidates
            if not (c["zone"] == override["zone"] and c["method"] == override["method"])
        ]
    debug_trace["modifier_applied"] = type_id or "none"

    # ── 최종 선택: score 기반 정렬 + zone 중복 제거 ──
    candidates.sort(key=lambda x: x["score"], reverse=True)

    visual_priorities: list[VisualPriority] = []
    recommended: list[RecommendedAction] = []
    seen_zones: set[str] = set()

    for c in candidates:
        if c["zone"] in seen_zones:
            continue
        seen_zones.add(c["zone"])

        if len(visual_priorities) < 3:
            visual_priorities.append(VisualPriority(
                zone=c["zone"], importance=round(c["score"], 2),
                reason=c["goal"], axis=c["axis"], delta_direction=c["direction"],
            ))
        if len(recommended) < MAX_RECOMMENDED_ACTIONS:
            recommended.append(RecommendedAction(
                zone=c["zone"], method=c["method"],
                goal=c["goal"], priority=len(recommended) + 1,
            ))

    # ── 최소 action 보장 (gap이 작거나 후보 부족 시) ──
    if len(recommended) < MIN_RECOMMENDED_ACTIONS:
        fallback_rules = action_rules.get(primary_axis, {}).get("increase", [])
        for rule in fallback_rules:
            if rule["zone"] not in seen_zones and len(recommended) < MIN_RECOMMENDED_ACTIONS:
                recommended.append(RecommendedAction(
                    zone=rule["zone"], method=rule["method"],
                    goal=rule["goal"], priority=len(recommended) + 1,
                ))
                seen_zones.add(rule["zone"])

    debug_trace["final_actions"] = [r.zone for r in recommended]

    # ── 기대 효과 ──
    type_label = matched_type.get("name_kr", "")
    style_tone = modifier.get("style_tone", "")
    expected_effects = _generate_expected_effects(primary_axis, type_label, style_tone)

    return ActionSpec(
        matched_type_id=type_id,
        matched_type_label=type_label,
        primary_gap_axis=primary_axis,
        visual_priorities=visual_priorities,
        recommended_actions=recommended,
        avoid_actions=avoid_actions,
        expected_effects=expected_effects,
        report_mode="grooming_male_v0" if gender == "male" else "makeup_female_v0",
        _debug_trace=debug_trace,
    )


# ─────────────────────────────────────────────
#  build_overlay_plan()
# ─────────────────────────────────────────────

def build_overlay_plan(
    action_spec: ActionSpec,
    face_features,
) -> list[OverlayZone]:
    """
    recommended_actions → 시각적 오버레이 계획.
    zone name(semantic)까지만 확정. landmark 변환은 렌더러에서 source_model 확인 후 수행.
    남성(grooming_male)은 오버레이 비활성 — 빈 리스트 반환.
    """
    if action_spec.report_mode.startswith("grooming"):
        return []
    modifier = get_type_modifier(action_spec.matched_type_id)
    intensity = modifier.get("shading_intensity", "medium")
    mult = {"minimal": 0.6, "light": 0.8, "medium": 1.0, "strong": 1.2}.get(intensity, 1.0)

    zones: list[OverlayZone] = []
    for action in action_spec.recommended_actions:
        visual = ZONE_VISUAL.get(action.zone)
        if not visual:
            continue
        ztype, color, opacity = visual
        zones.append(OverlayZone(
            zone_name=action.zone, zone_type=ztype,
            color_hex=color, opacity=min(opacity * mult, 0.5),
        ))
    return zones
