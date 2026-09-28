"""
SIGAK CV Pipeline — Face Analysis

InsightFace buffalo_l (primary) → MediaPipe FaceMesh (fallback)
→ 17개 구조적 특징 + 피부톤 분석

InsightFace: 106개 2D 랜드마크, 얼굴 전용 학습, 조명/각도 강건
MediaPipe: 468개 3D 랜드마크, 범용 모델, InsightFace 실패 시 폴백
"""
import logging
import math
import threading
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  FaceFeatures (하위 호환 유지)
# ─────────────────────────────────────────────

@dataclass
class FaceFeatures:
    """Extracted facial structure features."""
    face_shape: str              # oval, round, square, heart, oblong
    jaw_angle: float             # degrees
    cheekbone_prominence: float  # 0–1
    eye_width_ratio: float       # eye_w / face_w
    eye_spacing_ratio: float     # inner_eye_dist / face_w
    eye_ratio: float             # eye_h / eye_w (가로세로 비율)
    eye_tilt: float              # degrees (눈꼬리 기울기, +상향 -하향)
    nose_length_ratio: float
    nose_width_ratio: float
    nose_bridge_height: float    # 코 높이 (정규화)
    lip_fullness: float
    face_length_ratio: float     # face_h / face_w (종횡비)
    forehead_ratio: float
    brow_arch: float             # 눈썹 아치 높이 (정규화)
    brow_eye_distance: Optional[float]  # 눈썹-눈 거리 / face_height (가까울수록 강렬한 인상)
    philtrum_ratio: float        # 인중 길이 / 하안부 길이
    symmetry_score: float        # 0–1 (1 = perfect symmetry)
    golden_ratio_score: float    # proximity to phi ratios
    skin_tone: str               # cool / warm / neutral
    skin_brightness: float       # 0–1
    skin_warmth_score: float     # LAB 기반 raw warmth 값 (양수=warm, 음수=cool)
    skin_chroma: float           # LAB C* (채도) — undertone × chroma 분류용
    skin_hex_sample: str         # 볼 ROI 평균 피부색 hex (예: "#D4A574")
    landmarks: list              # raw landmarks as list of [x,y] or [x,y,z]
    landmarks_106: list          # InsightFace 106점 2D landmarks [[x,y], ...] 또는 빈 리스트
    bbox: list                   # [x1, y1, x2, y2] 또는 빈 리스트
    personal_color: Optional[dict] = None  # 4계절 퍼스널컬러 {season, subtype, label_kr, confidence}

    def to_dict(self):
        d = asdict(self)
        d.pop("landmarks")       # Too large for casual serialization
        d.pop("landmarks_106")   # overlay 렌더용 — to_dict에서 제외
        d.pop("bbox")            # overlay 렌더용
        return d


# ─────────────────────────────────────────────
#  공통 유틸리티
# ─────────────────────────────────────────────

def _dist2d(a, b):
    """2D 유클리드 거리."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _dist(a, b):
    """N차원 유클리드 거리 (2D/3D 모두 지원)."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _angle(a, b, c):
    """점 b에서의 a-b-c 각도 (도)."""
    ba = [ai - bi for ai, bi in zip(a, b)]
    bc = [ci - bi for ci, bi in zip(c, b)]
    dot = sum(x * y for x, y in zip(ba, bc))
    mag_ba = math.sqrt(sum(x ** 2 for x in ba))
    mag_bc = math.sqrt(sum(x ** 2 for x in bc))
    if mag_ba * mag_bc == 0:
        return 0
    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


# ── 얼굴형 분류 (v2: SCUT-FBP5500 캘리브레이션 기반) ──

# 캘리브레이션 범위 (SCUT 2000 + 셀럽 42장 합산)
# ratio max를 1.55로 확장 (셀럽에서 1.53까지 관측)
_FACE_CAL = {
    "ratio": (1.048, 1.55),
    "jaw":   (84.7, 131.8),
    "cheek": (0.336, 0.913),
    "fore":  (0.308, 0.530),
}

_OVAL_PENALTY = 0.90

_FACE_SHAPE_SIGNATURES = {
    # target: 정규화 공간(0~1)에서 해당 유형의 이상적 위치
    # weight: 해당 피처의 판별 중요도
    # 셀럽 42장 + SCUT 2000장 교차 검증 후 튜닝 (2026-04-10)
    "oval":              {"ratio": (0.55, 1.5), "jaw": (0.45, 1.5), "cheek": (0.50, 1.0), "fore": (0.50, 1.0)},
    "round":             {"ratio": (0.20, 2.0), "jaw": (0.80, 3.0), "cheek": (0.40, 0.5), "fore": (0.55, 0.5)},
    "oblong":            {"ratio": (0.95, 3.0), "jaw": (0.55, 0.5), "cheek": (0.45, 0.3), "fore": (0.50, 0.3)},
    "square":            {"ratio": (0.30, 1.5), "jaw": (0.10, 3.0), "cheek": (0.40, 1.0), "fore": (0.50, 0.5)},
    "heart":             {"ratio": (0.50, 0.5), "jaw": (0.25, 2.0), "cheek": (0.55, 1.0), "fore": (0.80, 3.0)},
    "inverted_triangle": {"ratio": (0.50, 0.5), "jaw": (0.15, 2.5), "cheek": (0.80, 3.0), "fore": (0.55, 1.0)},
    "diamond":           {"ratio": (0.55, 0.5), "jaw": (0.20, 1.5), "cheek": (0.90, 3.5), "fore": (0.20, 3.0)},
}


def _norm_feat(value: float, vmin: float, vmax: float) -> float:
    if vmax == vmin:
        return 0.5
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def _classify_face_shape(ratio, jaw_angle, cheekbone, forehead) -> str:
    """얼굴형 분류 (v2: 시그니처 유사도 + oval penalty)."""
    norms = {
        "ratio": _norm_feat(ratio, *_FACE_CAL["ratio"]),
        "jaw":   _norm_feat(jaw_angle, *_FACE_CAL["jaw"]),
        "cheek": _norm_feat(cheekbone, *_FACE_CAL["cheek"]),
        "fore":  _norm_feat(forehead, *_FACE_CAL["fore"]),
    }
    best_shape, best_score = "oval", -1.0
    for shape, sig in _FACE_SHAPE_SIGNATURES.items():
        w_sum, w_total = 0.0, 0.0
        for feat, (target, weight) in sig.items():
            w_sum += weight * (1.0 - abs(norms[feat] - target))
            w_total += weight
        score = w_sum / w_total if w_total > 0 else 0.0
        if shape == "oval":
            score *= _OVAL_PENALTY
        if score > best_score:
            best_score = score
            best_shape = shape
    return best_shape


def _analyze_skin_tone_from_image(image: np.ndarray, cheek_points: list[tuple]) -> dict:
    """
    얼굴 영역에서 피부톤을 분석한다.
    cheek_points: [(x, y), ...] 볼 영역 픽셀 좌표
    """
    h, w = image.shape[:2]
    skin_pixels = []
    sample_radius = max(5, int(w * 0.03))

    for cx, cy in cheek_points:
        cx, cy = int(cx), int(cy)
        y1, y2 = max(0, cy - sample_radius), min(h, cy + sample_radius)
        x1, x2 = max(0, cx - sample_radius), min(w, cx + sample_radius)
        region = image[y1:y2, x1:x2]
        if region.size > 0:
            skin_pixels.append(region.reshape(-1, 3))

    if not skin_pixels:
        return {
            "skin_tone": "neutral", "skin_brightness": 0.5,
            "skin_warmth_score": 0.0, "skin_chroma": 0.0, "skin_hex_sample": "#999999",
        }

    pixels = np.concatenate(skin_pixels, axis=0)
    pixels_bgr = pixels.reshape(1, -1, 3).astype(np.uint8)
    pixels_lab = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)

    l_mean = pixels_lab[:, 0].mean() / 255
    a_mean = pixels_lab[:, 1].mean()
    b_mean = pixels_lab[:, 2].mean()

    # warmth (v2: 아시안 피부 중심값 기준 z-score)
    # 아시안 피부의 전형적 warmth가 17~37 범위라 절대값 threshold(8)로는
    # 거의 전부 warm으로 분류됨. z-score로 상대 비교.
    _WARMTH_CENTER = 13.0  # 아시안 피부 warmth 중앙값 (셀럽 42장 실측: 13.2)
    _WARMTH_STD = 9.5      # warmth 표준편차 (셀럽 42장 실측: 9.5)
    warmth = (b_mean - 128) + (a_mean - 128) * 0.5
    warmth_z = (warmth - _WARMTH_CENTER) / _WARMTH_STD
    if warmth_z > 0.5:
        tone = "warm"
    elif warmth_z < -0.5:
        tone = "cool"
    else:
        tone = "neutral"

    # chroma (C*) — 채도 계산
    chroma = math.sqrt((a_mean - 128) ** 2 + (b_mean - 128) ** 2)

    # 볼 ROI 평균 BGR → hex (유저 피부색 샘플)
    avg_bgr = pixels.mean(axis=0)
    _c = lambda v: max(0, min(255, int(v)))
    hex_sample = "#{:02X}{:02X}{:02X}".format(
        _c(avg_bgr[2]), _c(avg_bgr[1]), _c(avg_bgr[0])  # BGR → RGB
    )

    # 4계절 퍼스널 컬러 분류 (v3)
    from pipeline.personal_color import PersonalColorClassifier

    pc_classifier = PersonalColorClassifier()
    pc_result = pc_classifier.classify(
        warmth=warmth,
        brightness=l_mean * 100,  # 0-1 -> 0-100
        chroma=chroma,
        is_calibrated=False,
    )

    return {
        "skin_tone": tone,
        "skin_brightness": round(float(l_mean), 3),
        "skin_warmth_score": round(float(warmth), 4),
        "skin_chroma": round(float(chroma), 1),
        "skin_hex_sample": hex_sample,
        "personal_color": {
            "season": pc_result.season,
            "subtype": pc_result.subtype,
            "label_kr": pc_result.label_kr,
            "confidence": pc_result.confidence,
        },
    }


# ═════════════════════════════════════════════
#  InsightFace 백엔드 (Primary)
# ═════════════════════════════════════════════

# InsightFace 3D-68 랜드마크 인덱스 (dlib 68-point 표준)
# 검증: buffalo_l landmark_3d_68 실 테스트로 구조 확인 (2026-04-06)
# 참조: https://ibug.doc.ic.ac.uk/resources/300-W/
INSIGHT_LM = {
    # 턱 컨투어 (0-16): 17개 점, 오른쪽→턱→왼쪽
    "jaw_right_start": 0,
    "jaw_right_corner": 4,   # 오른쪽 턱 모서리 (각도 계산용)
    "jaw_chin": 8,           # 턱 중앙 (가장 아래)
    "jaw_left_corner": 12,   # 왼쪽 턱 모서리
    "jaw_left_end": 16,

    # 오른쪽 눈썹 (17-21): subject 기준 오른쪽 = 이미지 왼쪽
    "right_brow_outer": 17,
    "right_brow_mid": 19,
    "right_brow_inner": 21,

    # 왼쪽 눈썹 (22-26)
    "left_brow_inner": 22,
    "left_brow_mid": 24,
    "left_brow_outer": 26,

    # 코 (27-35)
    "nose_bridge_top": 27,
    "nose_bridge_mid": 28,
    "nose_tip": 30,
    "nose_bottom": 33,       # 코 밑 중앙 (인중 계산용)
    "left_nostril": 35,
    "right_nostril": 31,

    # 오른쪽 눈 (36-41)
    "right_eye_outer": 36,
    "right_eye_top": 37,     # 상단 (37+38 평균 가능)
    "right_eye_inner": 39,
    "right_eye_bottom": 41,  # 하단 (40+41 평균 가능)

    # 왼쪽 눈 (42-47)
    "left_eye_inner": 42,
    "left_eye_top": 43,      # 상단 (43+44 평균 가능)
    "left_eye_outer": 45,
    "left_eye_bottom": 47,   # 하단 (46+47 평균 가능)

    # 입 외곽 (48-59)
    "mouth_right": 48,
    "upper_lip_top": 51,     # 윗입술 상단 중앙
    "mouth_left": 54,
    "lower_lip_bottom": 57,  # 아랫입술 하단 중앙
}

# InsightFace 싱글톤
_insight_app = None
_insight_lock = threading.Lock()


def _get_insight_app():
    """InsightFace FaceAnalysis 싱글톤."""
    global _insight_app
    if _insight_app is not None:
        return _insight_app

    with _insight_lock:
        if _insight_app is not None:
            return _insight_app
        try:
            from insightface.app import FaceAnalysis
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0, det_size=(640, 640))
            _insight_app = app
            logger.info("InsightFace buffalo_l 로드 완료")
        except Exception as e:
            logger.warning("InsightFace 로드 실패: %s", e)
            raise
    return _insight_app


def _analyze_with_insightface(image: np.ndarray) -> Optional[FaceFeatures]:
    """
    InsightFace landmark_3d_68 (dlib 68-point 표준) 기반 얼굴 분석.

    Args:
        image: BGR numpy 배열

    Returns:
        FaceFeatures or None
    """
    app = _get_insight_app()
    faces = app.get(image)
    if not faces:
        return None

    # 가장 큰 얼굴 선택
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    lm = face.landmark_3d_68
    if lm is None or lm.shape[0] < 68:
        return None

    # xy만 사용 (z는 3D 정보이나 2D 계산에 사용하지 않음)
    g = lambda key: lm[INSIGHT_LM[key]][:2].tolist()

    # ── 얼굴 크기 (bbox 기반) ──
    bbox = face.bbox
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    if face_width == 0 or face_height == 0:
        return None
    face_ratio = face_height / face_width

    # ── 턱선 각도 (4-8-12) ──
    jaw_l = g("jaw_left_corner")
    jaw_chin = g("jaw_chin")
    jaw_r = g("jaw_right_corner")
    jaw_angle = _angle(jaw_l, jaw_chin, jaw_r)

    # ── 광대 돌출도 ──
    jaw_w = _dist2d(jaw_l, jaw_r)
    cheekbone_prominence = min(1.0, max(0.0, (face_width - jaw_w) / face_width * 3))

    # ── 눈 지표 ──
    l_inner, l_outer = g("left_eye_inner"), g("left_eye_outer")
    r_inner, r_outer = g("right_eye_inner"), g("right_eye_outer")

    # 눈 높이: 상하 2개 점 평균 사용 (37+38, 40+41 등)
    r_top_y = (lm[37][1] + lm[38][1]) / 2
    r_bot_y = (lm[40][1] + lm[41][1]) / 2
    l_top_y = (lm[43][1] + lm[44][1]) / 2
    l_bot_y = (lm[46][1] + lm[47][1]) / 2

    left_eye_w = _dist2d(l_inner, l_outer)
    right_eye_w = _dist2d(r_inner, r_outer)
    avg_eye_w = (left_eye_w + right_eye_w) / 2
    eye_width_ratio = avg_eye_w / face_width

    eye_spacing = _dist2d(l_inner, r_inner)
    eye_spacing_ratio = eye_spacing / face_width

    left_eye_h = abs(l_top_y - l_bot_y)
    right_eye_h = abs(r_top_y - r_bot_y)
    avg_eye_h = (left_eye_h + right_eye_h) / 2
    eye_ratio = avg_eye_h / max(avg_eye_w, 0.001)

    # 눈꼬리 기울기 (outer-inner 벡터의 y 기울기)
    # dlib 68: R eye outer=36, inner=39 / L eye inner=42, outer=45
    r_tilt = math.degrees(math.atan2(
        lm[39][1] - lm[36][1], abs(lm[39][0] - lm[36][0])
    ))
    l_tilt = math.degrees(math.atan2(
        lm[42][1] - lm[45][1], abs(lm[45][0] - lm[42][0])
    ))
    eye_tilt = (r_tilt + l_tilt) / 2

    # ── 코 ──
    nose_bridge = g("nose_bridge_top")
    nose_tip = g("nose_tip")
    nose_length = _dist2d(nose_bridge, nose_tip)
    nose_length_ratio = nose_length / face_height

    l_nostril, r_nostril = g("left_nostril"), g("right_nostril")
    nose_width = _dist2d(l_nostril, r_nostril)
    nose_width_ratio = nose_width / face_width

    # 코 높이: z축 활용 (3D 68은 z 제공)
    nose_tip_z = float(lm[30][2])
    cheek_z = (float(lm[1][2]) + float(lm[15][2])) / 2  # 턱 컨투어 양쪽
    nose_bridge_height = abs(nose_tip_z - cheek_z) / max(face_height, 0.001)

    # ── 입술 ──
    upper_lip = g("upper_lip_top")
    lower_lip = g("lower_lip_bottom")
    lip_height = _dist2d(upper_lip, lower_lip)
    lip_fullness = lip_height / face_height

    # ── 얼굴 비율 ──
    face_length_ratio = face_ratio

    # ── 이마 비율 ──
    forehead_h = nose_bridge[1] - bbox[1]
    forehead_ratio = max(0, forehead_h / face_height)

    # ── 눈썹 아치 ──
    l_brow_mid = g("left_brow_mid")
    l_brow_base_y = (g("left_brow_inner")[1] + g("left_brow_outer")[1]) / 2
    r_brow_mid = g("right_brow_mid")
    r_brow_base_y = (g("right_brow_inner")[1] + g("right_brow_outer")[1]) / 2
    left_arch = abs(l_brow_mid[1] - l_brow_base_y)
    right_arch = abs(r_brow_mid[1] - r_brow_base_y)
    brow_arch = ((left_arch + right_arch) / 2) / face_height

    # ── 눈썹-눈 거리 (106점 2D 랜드마크 기반) ──
    brow_eye_distance_val: Optional[float] = None
    try:
        _lm106 = face.landmark_2d_106
        if _lm106 is not None and _lm106.shape[0] >= 93:
            # InsightFace 106점: 오른쪽 눈썹 [38-42], 오른쪽 눈 [33-37]
            r_brow_pts = _lm106[38:43]  # indices 38, 39, 40, 41, 42
            r_eye_pts = _lm106[33:38]   # indices 33, 34, 35, 36, 37
            r_brow_center_y = float(r_brow_pts[:, 1].mean())
            r_eye_center_y = float(r_eye_pts[:, 1].mean())
            brow_eye_distance_val = abs(r_brow_center_y - r_eye_center_y) / face_height
            brow_eye_distance_val = round(brow_eye_distance_val, 4)
    except Exception:
        brow_eye_distance_val = None

    # ── 인중 비율 ──
    nose_bottom = g("nose_bottom")
    chin = g("jaw_chin")
    philtrum_length = _dist2d(nose_bottom, upper_lip)
    lower_face = _dist2d(nose_bottom, chin)
    philtrum_ratio = philtrum_length / max(lower_face, 0.001)

    # ── 대칭도 ──
    face_cx = (bbox[0] + bbox[2]) / 2
    pairs = [
        (l_inner, r_inner),
        (l_outer, r_outer),
        (g("mouth_left"), g("mouth_right")),
        (g("left_brow_outer"), g("right_brow_outer")),
        (jaw_l, jaw_r),
    ]
    sym_diffs = []
    for lp, rp in pairs:
        dl = abs(lp[0] - face_cx)
        dr = abs(rp[0] - face_cx)
        sym_diffs.append(1 - abs(dl - dr) / max(dl, dr, 0.001))
    symmetry_score = sum(sym_diffs) / len(sym_diffs)

    # ── 황금비 ──
    phi = 1.618
    ratios = [face_ratio, forehead_ratio * 3, eye_spacing_ratio * 5]
    phi_diffs = [1 - min(abs(r - phi) / phi, 1) for r in ratios]
    golden_ratio_score = sum(phi_diffs) / len(phi_diffs)

    # ── 얼굴형 ──
    face_shape = _classify_face_shape(face_ratio, jaw_angle, cheekbone_prominence, forehead_ratio)

    # ── 피부톤 (턱 컨투어 볼 영역) ──
    cheek_r_pt = lm[2][:2]  # 오른쪽 볼
    cheek_l_pt = lm[14][:2]  # 왼쪽 볼
    skin = _analyze_skin_tone_from_image(image, [
        (int(cheek_r_pt[0]), int(cheek_r_pt[1])),
        (int(cheek_l_pt[0]), int(cheek_l_pt[1])),
    ])

    # ── 랜드마크 패키징 (68개, xyz) ──
    raw_landmarks = [[float(x), float(y), float(z)] for x, y, z in lm]

    # ── 106점 2D + bbox (overlay 렌더용) ──
    lm106 = face.landmark_2d_106
    raw_106 = [[float(x), float(y)] for x, y in lm106] if lm106 is not None else []
    raw_bbox = [float(v) for v in bbox] if bbox is not None else []

    return FaceFeatures(
        face_shape=face_shape,
        jaw_angle=round(jaw_angle, 1),
        cheekbone_prominence=round(cheekbone_prominence, 3),
        eye_width_ratio=round(eye_width_ratio, 3),
        eye_spacing_ratio=round(eye_spacing_ratio, 3),
        eye_ratio=round(eye_ratio, 3),
        eye_tilt=round(eye_tilt, 2),
        nose_length_ratio=round(nose_length_ratio, 3),
        nose_width_ratio=round(nose_width_ratio, 3),
        nose_bridge_height=round(nose_bridge_height, 3),
        lip_fullness=round(lip_fullness, 4),
        face_length_ratio=round(float(face_length_ratio), 3),
        forehead_ratio=round(forehead_ratio, 3),
        brow_arch=round(brow_arch, 4),
        brow_eye_distance=brow_eye_distance_val,
        philtrum_ratio=round(philtrum_ratio, 3),
        symmetry_score=round(symmetry_score, 3),
        golden_ratio_score=round(golden_ratio_score, 3),
        skin_tone=skin["skin_tone"],
        skin_brightness=skin["skin_brightness"],
        skin_warmth_score=skin["skin_warmth_score"],
        skin_chroma=skin["skin_chroma"],
        skin_hex_sample=skin["skin_hex_sample"],
        personal_color=skin.get("personal_color"),
        landmarks=raw_landmarks,
        landmarks_106=raw_106,
        bbox=raw_bbox,
    )


# ═════════════════════════════════════════════
#  MediaPipe 백엔드 (Fallback, lazy import)
# ═════════════════════════════════════════════

# MediaPipe 468 랜드마크 인덱스
MP_LM = {
    "forehead_top": 10, "chin_bottom": 152,
    "left_cheek": 234, "right_cheek": 454,
    "nose_tip": 1, "nose_bridge": 6,
    "left_eye_inner": 133, "left_eye_outer": 33,
    "right_eye_inner": 362, "right_eye_outer": 263,
    "left_jaw": 172, "right_jaw": 397, "jaw_tip": 152,
    "left_lip_corner": 61, "right_lip_corner": 291,
    "upper_lip": 13, "lower_lip": 14,
    "left_brow_inner": 107, "left_brow_outer": 70,
    "right_brow_inner": 336, "right_brow_outer": 300,
    "left_eye_top": 159, "left_eye_bottom": 145,
    "right_eye_top": 386, "right_eye_bottom": 374,
    "nose_bottom": 2,
    "left_brow_mid": 105, "right_brow_mid": 334,
}


def _analyze_with_mediapipe(image: np.ndarray) -> Optional[FaceFeatures]:
    """MediaPipe FaceMesh 468 랜드마크 기반 얼굴 분석 (폴백)."""
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    g = lambda idx: [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
    h, w = image.shape[:2]

    # 얼굴 크기 (정규화 좌표 기반)
    face_height = _dist(g(MP_LM["forehead_top"]), g(MP_LM["chin_bottom"]))
    face_width = _dist(g(MP_LM["left_cheek"]), g(MP_LM["right_cheek"]))
    if face_height == 0 or face_width == 0:
        return None
    face_ratio = face_height / face_width

    # 턱선 각도
    jaw_angle = _angle(g(MP_LM["left_jaw"]), g(MP_LM["jaw_tip"]), g(MP_LM["right_jaw"]))

    # 광대
    jaw_w = _dist(g(MP_LM["left_jaw"]), g(MP_LM["right_jaw"]))
    cheekbone_prominence = min(1.0, max(0.0, (face_width - jaw_w) / face_width * 3))

    # 눈
    l_inner, l_outer = g(MP_LM["left_eye_inner"]), g(MP_LM["left_eye_outer"])
    r_inner, r_outer = g(MP_LM["right_eye_inner"]), g(MP_LM["right_eye_outer"])
    left_eye_w = _dist(l_inner, l_outer)
    right_eye_w = _dist(r_inner, r_outer)
    avg_eye_w = (left_eye_w + right_eye_w) / 2
    eye_width_ratio = avg_eye_w / face_width
    eye_spacing = _dist(l_inner, r_inner)
    eye_spacing_ratio = eye_spacing / face_width

    l_top, l_bot = g(MP_LM["left_eye_top"]), g(MP_LM["left_eye_bottom"])
    r_top, r_bot = g(MP_LM["right_eye_top"]), g(MP_LM["right_eye_bottom"])
    avg_eye_h = (_dist(l_top, l_bot) + _dist(r_top, r_bot)) / 2
    eye_ratio = avg_eye_h / max(avg_eye_w, 0.001)

    left_tilt = math.degrees(math.atan2(l_outer[1] - l_inner[1], abs(l_outer[0] - l_inner[0])))
    right_tilt = math.degrees(math.atan2(r_inner[1] - r_outer[1], abs(r_inner[0] - r_outer[0])))
    eye_tilt = (left_tilt + right_tilt) / 2

    # 코
    nose_length = _dist(g(MP_LM["nose_bridge"]), g(MP_LM["nose_tip"]))
    nose_length_ratio = nose_length / face_height
    nose_width_ratio = 0.25  # MediaPipe에서 콧볼 추정 어려움

    nose_tip_z = g(MP_LM["nose_tip"])[2]
    cheek_z = (g(MP_LM["left_cheek"])[2] + g(MP_LM["right_cheek"])[2]) / 2
    nose_bridge_height = abs(nose_tip_z - cheek_z) / max(face_height, 0.001)

    # 입술
    lip_height = _dist(g(MP_LM["upper_lip"]), g(MP_LM["lower_lip"]))
    lip_fullness = lip_height / face_height

    face_length_ratio = face_ratio

    # 이마
    forehead_h = _dist(g(MP_LM["forehead_top"]), g(MP_LM["nose_bridge"]))
    forehead_ratio = forehead_h / face_height

    # 눈썹 아치
    l_brow_mid = g(MP_LM["left_brow_mid"])
    l_brow_base_y = (g(MP_LM["left_brow_inner"])[1] + g(MP_LM["left_brow_outer"])[1]) / 2
    r_brow_mid = g(MP_LM["right_brow_mid"])
    r_brow_base_y = (g(MP_LM["right_brow_inner"])[1] + g(MP_LM["right_brow_outer"])[1]) / 2
    brow_arch = ((abs(l_brow_mid[1] - l_brow_base_y) + abs(r_brow_mid[1] - r_brow_base_y)) / 2) / face_height

    # 인중
    nose_bottom = g(MP_LM["nose_bottom"])
    upper_lip = g(MP_LM["upper_lip"])
    chin = g(MP_LM["chin_bottom"])
    philtrum_length = _dist(nose_bottom, upper_lip)
    lower_face = _dist(nose_bottom, chin)
    philtrum_ratio = philtrum_length / max(lower_face, 0.001)

    # 대칭도
    center = g(MP_LM["nose_tip"])
    sym_pairs = [
        (MP_LM["left_eye_outer"], MP_LM["right_eye_outer"]),
        (MP_LM["left_eye_inner"], MP_LM["right_eye_inner"]),
        (MP_LM["left_lip_corner"], MP_LM["right_lip_corner"]),
        (MP_LM["left_brow_outer"], MP_LM["right_brow_outer"]),
        (MP_LM["left_jaw"], MP_LM["right_jaw"]),
    ]
    sym_diffs = []
    for l_idx, r_idx in sym_pairs:
        dl = _dist(g(l_idx), center)
        dr = _dist(g(r_idx), center)
        sym_diffs.append(1 - abs(dl - dr) / max(dl, dr, 0.001))
    symmetry_score = sum(sym_diffs) / len(sym_diffs)

    # 황금비
    phi = 1.618
    ratios = [face_ratio, forehead_ratio * 3, eye_spacing_ratio * 5]
    phi_diffs = [1 - min(abs(r - phi) / phi, 1) for r in ratios]
    golden_ratio_score = sum(phi_diffs) / len(phi_diffs)

    face_shape = _classify_face_shape(face_ratio, jaw_angle, cheekbone_prominence, forehead_ratio)

    # 피부톤 (MediaPipe 정규화좌표 → 픽셀 변환)
    cheek_pts = [
        (int(landmarks[50].x * w), int(landmarks[50].y * h)),
        (int(landmarks[280].x * w), int(landmarks[280].y * h)),
    ]
    skin = _analyze_skin_tone_from_image(image, cheek_pts)

    raw_landmarks = [[lm.x, lm.y, lm.z] for lm in landmarks]

    return FaceFeatures(
        face_shape=face_shape,
        jaw_angle=round(jaw_angle, 1),
        cheekbone_prominence=round(cheekbone_prominence, 3),
        eye_width_ratio=round(eye_width_ratio, 3),
        eye_spacing_ratio=round(eye_spacing_ratio, 3),
        eye_ratio=round(eye_ratio, 3),
        eye_tilt=round(eye_tilt, 2),
        nose_length_ratio=round(nose_length_ratio, 3),
        nose_width_ratio=nose_width_ratio,
        nose_bridge_height=round(nose_bridge_height, 3),
        lip_fullness=round(lip_fullness, 4),
        face_length_ratio=round(face_length_ratio, 3),
        forehead_ratio=round(forehead_ratio, 3),
        brow_arch=round(brow_arch, 4),
        brow_eye_distance=None,  # MediaPipe에서는 106점 기반 계산 불가
        philtrum_ratio=round(philtrum_ratio, 3),
        symmetry_score=round(symmetry_score, 3),
        golden_ratio_score=round(golden_ratio_score, 3),
        skin_tone=skin["skin_tone"],
        skin_brightness=skin["skin_brightness"],
        skin_warmth_score=skin["skin_warmth_score"],
        skin_chroma=skin["skin_chroma"],
        skin_hex_sample=skin["skin_hex_sample"],
        personal_color=skin.get("personal_color"),
        landmarks=raw_landmarks,
        landmarks_106=[],  # MediaPipe에는 106점 없음
        bbox=[],
    )


# ═════════════════════════════════════════════
#  메인 진입점 — InsightFace 우선, MediaPipe 폴백
# ═════════════════════════════════════════════

def analyze_face(image_bytes: bytes) -> Optional[FaceFeatures]:
    """
    얼굴 분석 메인 함수.

    InsightFace (buffalo_l 106 랜드마크) 전용.
    실패 시 None 반환 → 호출부에서 에러 처리.

    Args:
        image_bytes: JPEG/PNG 이미지 바이트

    Returns:
        FaceFeatures or None (얼굴 미검출)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None

    # InsightFace 전용 — MediaPipe 폴백 비활성
    # MediaPipe는 106점 랜드마크/brow_eye_distance/overlay가 불가하여
    # 품질 저하된 결과를 내보내는 것보다 실패 반환이 나음
    try:
        result = _analyze_with_insightface(image)
        if result is not None:
            return result
        logger.info("InsightFace 얼굴 미검출 — 반환 None")
    except Exception as e:
        logger.warning("InsightFace 실패: %s", e)

    return None
