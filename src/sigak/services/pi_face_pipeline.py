"""SIGAK PI-B — 정면 raw 1장 → PiFaceResult 단일 entry point.

PI v1 진입 흐름 ("완벽한 분석을 위해서는 ㅇㅇ님 정면 사진 한 장이 필요해요" → 사진 업로드)
직후 호출되는 face / CLIP / anchor 영역. PI-A (services/pi_engine_v3.py 신규)
가 이 함수를 호출해 9 컴포넌트 데이터 조립에 활용.

CLAUDE.md PI v1 컴포넌트 매핑:
  - face-structure (raw 강)        ← face_features + face_shape
  - coordinate-map (raw 강)        ← coord_3axis (VisualCoordinate, 0~1 외부)
  - celeb-reference (raw 강)       ← matched_celebs (top-3)
  - type-reference (vault 강)      ← matched_types (top-3, vault 와 결합)

PI-A 인터페이스 spec (2026-04-25 확정):
  face_features  — dict, 18 numeric metrics (face_shape 등 categorical 제외)
  coord_3axis    — VisualCoordinate (0~1 외부 스케일)
  matched_celebs — list[dict] {celeb_name, similarity, rank}
  matched_types  — list[dict] {type_id, name_kr, similarity, rank,
                                coords{shape,volume,age}, delta_axis, delta_note}

LLM 격리 (PII 보호):
  CLIP embedding 은 prompt 에 절대 안 들어감. R2 분리 보존만.
  PI-A 가 LLM payload 만들 때는 PiFaceResult.to_prompt_safe_dict() 사용.

R2 영구 보존 (product DNA):
  user_media/{uid}/pi_baseline/{ts}.jpg     — 원본 정면 사진
  user_media/{uid}/pi_embedding/{ts}.npy    — 768d L2-normalized 임베딩
  user_media/{uid}/pi_landmarks/{ts}.json   — InsightFace 106 랜드마크 + bbox
"""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import cv2
import numpy as np

from services.coordinate_system import VisualCoordinate


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  PI-A 인터페이스 — 18 face features
# ─────────────────────────────────────────────

#: PI-A 가 face_features 에서 기대하는 numeric metric 키 (18개).
#: face_shape (categorical) / skin_tone (categorical) / brow_eye_distance / skin_chroma /
#: skin_hex_sample / personal_color 는 제외 — 별도 face_shape 필드 또는 PI-A 미사용.
FACE_FEATURE_KEYS: tuple[str, ...] = (
    "jaw_angle", "cheekbone_prominence",
    "eye_width_ratio", "eye_spacing_ratio", "eye_ratio", "eye_tilt",
    "nose_length_ratio", "nose_width_ratio", "nose_bridge_height", "lip_fullness",
    "forehead_ratio", "brow_arch", "philtrum_ratio",
    "symmetry_score", "golden_ratio_score",
    "skin_brightness", "skin_warmth_score",
    "face_length_ratio",
)


# ─────────────────────────────────────────────
#  예외
# ─────────────────────────────────────────────

class PIFaceError(Exception):
    """PI-B 단계 복구 불가 오류 — caller (PI-A) 가 토큰 환불 / UI 안내 결정."""


# ─────────────────────────────────────────────
#  결과 데이터
# ─────────────────────────────────────────────

@dataclass
class PiFaceResult:
    """PI-B 통합 출력. PI-A 가 9 컴포넌트 조립 시 raw 강 영역 데이터 소스.

    스키마 = PI-A 인터페이스 spec (2026-04-25 확정).
    """

    user_id: str
    gender: Literal["female", "male"]
    snapshot_ts: str                          # YYYYMMDDTHHMMSSZ — R2 키 일부

    # ── face.py 산출
    face_features: dict[str, float]           # 18 numeric metrics (FACE_FEATURE_KEYS)
    face_shape: str                           # categorical — oval / round / heart / ...

    # ── coordinate.py 산출 (0~1 외부 스케일 Pydantic)
    coord_3axis: VisualCoordinate

    # ── CLIP (LLM 격리)
    clip_embedding: np.ndarray                # 768d float32 L2-normalized

    # ── anchor matching (PI-A spec 형태)
    matched_types: list[dict] = field(default_factory=list)
    # 각: {type_id, name_kr, similarity, rank, coords{shape,volume,age}, delta_axis, delta_note}
    matched_celebs: list[dict] = field(default_factory=list)
    # 각: {celeb_name, similarity, rank}

    # ── R2 영구 보존 키 (persist_to_r2=False 거나 실패 시 None)
    baseline_photo_r2_key: Optional[str] = None
    clip_embedding_r2_key: Optional[str] = None
    landmarks_r2_key: Optional[str] = None

    def to_prompt_safe_dict(self) -> dict:
        """LLM payload 용 직렬화 — CLIP embedding 제외, R2 키만 노출."""
        return {
            "gender": self.gender,
            "snapshot_ts": self.snapshot_ts,
            "face_features": dict(self.face_features),
            "face_shape": self.face_shape,
            "coord_3axis": self.coord_3axis.model_dump(mode="json"),
            "matched_types": list(self.matched_types),
            "matched_celebs": list(self.matched_celebs),
            "r2_keys": {
                "baseline_photo": self.baseline_photo_r2_key,
                "clip_embedding": self.clip_embedding_r2_key,
                "landmarks": self.landmarks_r2_key,
            },
        }


# ─────────────────────────────────────────────
#  Main entry
# ─────────────────────────────────────────────

def analyze_baseline_photo(
    photo_bytes: bytes,
    *,
    gender: Literal["female", "male"],
    user_id: str,
    persist_to_r2: bool = True,
    snapshot_ts: Optional[str] = None,
    top_k_types: int = 3,
    top_k_celebs: int = 3,
    embedder=None,                            # 테스트용 dependency injection
) -> PiFaceResult:
    """정면 raw 1장 → PiFaceResult.

    Args:
      photo_bytes: 정면 사진 (JPEG/PNG/WebP 등 OpenCV 디코드 가능 포맷)
      gender: female | male — calibration_3axis.yaml 섹션 결정
      user_id: R2 키 prefix
      persist_to_r2: True 면 baseline + embedding + landmarks R2 영구 저장 시도.
        실패해도 해당 *_r2_key 만 None 으로 남고 결과는 정상 반환 (graceful degrade).
      snapshot_ts: 테스트 결정성 위해 외부 주입 가능. None 이면 utcnow 자동.
      top_k_types / top_k_celebs: 매칭 상위 N
      embedder: CLIPEmbedder 인스턴스 주입 (테스트용 mock). None 이면 싱글톤 로드.

    Raises:
      PIFaceError —
        - 이미지 디코드 실패
        - 얼굴 미검출 (analyze_face None)
        - CLIP 임베딩 산출 실패
        - 좌표 계산 실패 (CoordinateConfigError)
    """
    # 0. ts
    if snapshot_ts is None:
        snapshot_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # 1. 디코드
    try:
        nparr = np.frombuffer(photo_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise PIFaceError(f"image decode failed: {e}")
    if image_bgr is None or image_bgr.size == 0:
        raise PIFaceError("image decode returned empty")

    # 2. face.py — full features + landmarks + bbox
    from pipeline.face import analyze_face
    face_result = analyze_face(photo_bytes)
    if face_result is None:
        raise PIFaceError("face not detected (InsightFace)")

    face_full = face_result.to_dict()         # landmarks 등 제외된 dict
    face_shape = str(face_full.get("face_shape", "unknown"))
    face_features = _filter_face_features(face_full)

    # 3. CLIP 768d (얼굴 크롭 적용)
    embedding = _compute_clip_embedding(
        image_bgr=image_bgr,
        face_result=face_result,
        embedder=embedder,
    )

    # 4. 3축 좌표 — internal -1~+1 산출 → VisualCoordinate (0~1 외부)
    coords_internal = _compute_coords_safe(face_full, gender)
    coord_3axis = VisualCoordinate.from_internal(
        shape=float(coords_internal.get("shape", 0.0)),
        volume=float(coords_internal.get("volume", 0.0)),
        age=float(coords_internal.get("age", 0.0)),
    )

    # 5. 유형 매칭 — similarity.py 재활용 후 PI-A spec 형태로 변환
    from pipeline.similarity import find_similar_types
    types_raw = find_similar_types(
        user_embedding=embedding,
        user_coords=coords_internal,          # similarity 는 내부 스케일 -1~+1 기대
        gender=gender,
        top_k=top_k_types,
    )
    matched_types = _format_matched_types(types_raw, user_external=coord_3axis)

    # 6. 셀럽 매칭 — celeb_similarity.py + PI-A spec 형태로 변환
    from pipeline.celeb_similarity import find_similar_celebs
    celebs_raw = find_similar_celebs(
        user_embedding=embedding,
        gender=gender,
        top_k=top_k_celebs,
    )
    matched_celebs = _format_matched_celebs(celebs_raw)

    # 7. R2 영구 보존 (옵션 + best-effort)
    r2_baseline_key: Optional[str] = None
    r2_embedding_key: Optional[str] = None
    r2_landmarks_key: Optional[str] = None
    if persist_to_r2:
        r2_baseline_key = _persist_baseline_photo(
            photo_bytes=photo_bytes, user_id=user_id, snapshot_ts=snapshot_ts,
        )
        r2_embedding_key = _persist_embedding(
            embedding=embedding, user_id=user_id, snapshot_ts=snapshot_ts,
        )
        r2_landmarks_key = _persist_landmarks(
            face_result=face_result, user_id=user_id, snapshot_ts=snapshot_ts,
        )

    return PiFaceResult(
        user_id=user_id,
        gender=gender,
        snapshot_ts=snapshot_ts,
        face_features=face_features,
        face_shape=face_shape,
        coord_3axis=coord_3axis,
        clip_embedding=embedding,
        matched_types=matched_types,
        matched_celebs=matched_celebs,
        baseline_photo_r2_key=r2_baseline_key,
        clip_embedding_r2_key=r2_embedding_key,
        landmarks_r2_key=r2_landmarks_key,
    )


# ─────────────────────────────────────────────
#  Internal: feature filter + scale convert
# ─────────────────────────────────────────────

def _filter_face_features(full_dict: dict) -> dict[str, float]:
    """FaceFeatures.to_dict() → PI-A spec 18 numeric keys 만 추출.

    None / 비수치 값은 제외 (downstream 모델이 NaN 으로 깨지는 사고 방지).
    """
    out: dict[str, float] = {}
    for k in FACE_FEATURE_KEYS:
        v = full_dict.get(k)
        if v is None:
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _to_external_scalar(internal: float) -> float:
    """단일 축 -1~+1 (내부) → 0~1 (외부)."""
    return max(0.0, min(1.0, (float(internal) + 1.0) / 2.0))


# ─────────────────────────────────────────────
#  Internal: matched_types / matched_celebs 변환
# ─────────────────────────────────────────────

def _format_matched_types(
    raw: list[dict],
    *,
    user_external: VisualCoordinate,
) -> list[dict]:
    """find_similar_types 결과 → PI-A spec dict.

    각 항목:
      type_id, name_kr, similarity, rank,
      coords{shape, volume, age}      — 외부 0~1
      delta_axis{shape, volume, age}  — user_external - type_external (-1 ~ +1)
      delta_note                      — 가장 큰 |delta| 축의 방향 짧은 한글 서술
    """
    from pipeline.similarity import load_anchors

    anchors = load_anchors().get("anchors", {})
    out: list[dict] = []

    for rank, t in enumerate(raw, start=1):
        key = t.get("key")
        info = anchors.get(key, {})
        ref_internal = info.get("coords", {}) or {}

        type_external = {
            ax: round(_to_external_scalar(ref_internal.get(ax, 0.0)), 4)
            for ax in ("shape", "volume", "age")
        }

        # delta = user (0~1) - type (0~1)  → range [-1, +1], 양수=유저가 더 high 축
        ue = {"shape": user_external.shape, "volume": user_external.volume, "age": user_external.age}
        delta_axis = {
            ax: round(ue[ax] - type_external[ax], 4)
            for ax in ("shape", "volume", "age")
        }

        out.append({
            "type_id": info.get("type_id", t.get("type_id")),
            "name_kr": info.get("name_kr", t.get("name_kr")),
            "similarity": t.get("similarity"),
            "rank": rank,
            "coords": type_external,
            "delta_axis": delta_axis,
            "delta_note": _build_delta_note(delta_axis),
        })

    return out


def _format_matched_celebs(raw_celebs: list) -> list[dict]:
    """celeb_similarity.find_similar_celebs(CelebMatch list) → PI-A spec dict.

    각 항목 = {celeb_name, similarity, rank}. 그 외 키 (key/name_en/group) 는
    PI-A 가 요구 안 함 — 미노출.
    """
    out: list[dict] = []
    for rank, m in enumerate(raw_celebs, start=1):
        out.append({
            "celeb_name": m.name_kr,
            "similarity": m.similarity,
            "rank": rank,
        })
    return out


def _build_delta_note(delta_axis: dict[str, float]) -> str:
    """가장 큰 |delta| 축 1개를 짧은 한글 표기.

    |delta| < 0.05 → "거의 일치"
    그 외      → "{축_kr} {방향}쪽으로 {abs(delta):.2f}"

    예:
      {"shape": 0.12, "volume": -0.02, "age": 0.04}  → "shape 샤프쪽으로 0.12"
    """
    if not delta_axis:
        return ""
    primary_axis, primary_val = max(
        delta_axis.items(), key=lambda kv: abs(kv[1]),
    )
    if abs(primary_val) < 0.05:
        return "거의 일치"

    # 축별 방향 한글
    direction_map = {
        "shape":  ("소프트", "샤프"),
        "volume": ("평면",   "입체"),
        "age":    ("프레시", "성숙"),
    }
    neg, pos = direction_map.get(primary_axis, ("", ""))
    direction_kr = pos if primary_val > 0 else neg
    return f"{primary_axis} {direction_kr}쪽으로 {abs(primary_val):.2f}"


# ─────────────────────────────────────────────
#  Internal: CLIP helper
# ─────────────────────────────────────────────

def _compute_clip_embedding(
    *,
    image_bgr: np.ndarray,
    face_result,                 # pipeline.face.FaceFeatures
    embedder,
) -> np.ndarray:
    """얼굴 영역 크롭 후 CLIPEmbedder 로 768d 임베딩 산출.

    crop 실패 시 전체 이미지로 fallback (성능 저하 가능 — 로그).
    """
    from pipeline.clip import CLIPEmbedder, crop_face

    if embedder is None:
        embedder = CLIPEmbedder()

    cropped: Optional[np.ndarray] = None
    landmarks = getattr(face_result, "landmarks", None) or []
    if landmarks:
        try:
            cropped = crop_face(image_bgr, landmarks, padding=0.3)
            if cropped is None or cropped.size == 0:
                cropped = None
        except Exception:
            logger.exception("crop_face 실패 — 전체 이미지로 fallback")

    target = cropped if cropped is not None else image_bgr

    try:
        emb = embedder.extract(target)
    except Exception as e:
        raise PIFaceError(f"CLIP embedding failed: {e}")

    if emb is None or emb.ndim != 1 or emb.shape[0] != 768:
        raise PIFaceError(
            f"CLIP embedding malformed: shape={None if emb is None else emb.shape}"
        )
    norm = float(np.linalg.norm(emb))
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


def _compute_coords_safe(face_metrics: dict, gender: str) -> dict[str, float]:
    """compute_coordinates 호출 — 미캘리브레이션 시 PIFaceError 변환.

    출력: 내부 스케일 -1~+1 dict (VisualCoordinate.from_internal 입력용).
    """
    from pipeline.coordinate import (
        CoordinateConfigError,
        compute_coordinates,
    )
    try:
        return compute_coordinates(face_metrics, gender=gender)
    except CoordinateConfigError as e:
        raise PIFaceError(f"coordinate config error ({gender}): {e}")
    except Exception as e:
        raise PIFaceError(f"coordinate compute failed: {e}")


# ─────────────────────────────────────────────
#  R2 persistence (best-effort, 실패해도 결과는 반환)
# ─────────────────────────────────────────────

def _persist_baseline_photo(
    *, photo_bytes: bytes, user_id: str, snapshot_ts: str,
) -> Optional[str]:
    """원본 정면 사진 R2 저장. 키: user_media/{uid}/pi_baseline/{ts}.jpg"""
    try:
        from services import r2_client
        suffix = f"pi_baseline/{snapshot_ts}.jpg"
        key = r2_client.user_photo_key(user_id, suffix)
        r2_client.put_bytes(key, photo_bytes, content_type="image/jpeg")
        return key
    except Exception:
        logger.exception(
            "PI-B baseline photo R2 put failed: user=%s ts=%s", user_id, snapshot_ts,
        )
        return None


def _persist_embedding(
    *, embedding: np.ndarray, user_id: str, snapshot_ts: str,
) -> Optional[str]:
    """768d 임베딩 R2 저장 (.npy). 키: user_media/{uid}/pi_embedding/{ts}.npy"""
    try:
        from services import r2_client
        buf = io.BytesIO()
        np.save(buf, embedding.astype(np.float32), allow_pickle=False)
        body = buf.getvalue()
        suffix = f"pi_embedding/{snapshot_ts}.npy"
        key = r2_client.user_photo_key(user_id, suffix)
        r2_client.put_bytes(key, body, content_type="application/octet-stream")
        return key
    except Exception:
        logger.exception(
            "PI-B embedding R2 put failed: user=%s ts=%s", user_id, snapshot_ts,
        )
        return None


def _persist_landmarks(
    *, face_result, user_id: str, snapshot_ts: str,
) -> Optional[str]:
    """InsightFace 106 랜드마크 + bbox 를 JSON 으로 R2 저장.

    PI-A 의 face-structure / overlay 컴포넌트가 활용 가능.
    """
    try:
        from services import r2_client
        payload = {
            "snapshot_ts": snapshot_ts,
            "landmarks_106": getattr(face_result, "landmarks_106", []),
            "bbox": getattr(face_result, "bbox", []),
            # 468 raw 는 용량 크고 PI-A 가 거의 안 씀 — 보류 (필요 시 추가)
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        suffix = f"pi_landmarks/{snapshot_ts}.json"
        key = r2_client.user_photo_key(user_id, suffix)
        r2_client.put_bytes(key, body, content_type="application/json")
        return key
    except Exception:
        logger.exception(
            "PI-B landmarks R2 put failed: user=%s ts=%s", user_id, snapshot_ts,
        )
        return None
