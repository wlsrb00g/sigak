"""
SIGAK LLM Pipeline — Interview Interpretation + Report Generation

Uses Claude API to:
1. Parse natural language aspirations → aspiration coordinates
2. Generate personalized PI report narrative
"""
import contextlib
import contextvars
import json
from datetime import datetime, timezone
from typing import Iterator, Optional
import anthropic

from config import get_settings
from pipeline.similarity import get_type_reference_prompt, load_anchors
from pipeline.face_comparison import format_comparison_for_report
from pipeline.cluster import format_cluster_for_report


# ─────────────────────────────────────────────
#  LLM raw capture (데이터 기업 원칙 — LLM 출력 영구 보존)
# ─────────────────────────────────────────────
#
# 옛 SIGAK_V3 PI 는 _call_llm 으로 5+ 호출 (face / interview / type_match /
# gap_narration / finale). 각 raw 응답은 재현 불가능한 LLM 출력이라 영구
# 보존 대상. caller 가 with llm_raw_capture() 로 감싸면 그 안의 모든 _call_llm
# 호출의 raw text 가 list 에 자동 누적됨. 끝나면 caller 가 R2 dead-letter aware
# put 으로 영구 저장.

_llm_raw_capture: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "_llm_raw_capture", default=None,
)


@contextlib.contextmanager
def llm_raw_capture() -> Iterator[list[dict]]:
    """이 컨텍스트 안의 _call_llm raw 응답을 list 에 누적.

    Yield 되는 list 의 각 항목:
      {"captured_at": ISO8601, "raw": "<response text>", "system_excerpt": "...",
       "input_excerpt": "..."}

    caller 예시:
      with llm_raw_capture() as raws:
          interpret_face_structure(...)
          interpret_interview(...)
      # raws 에 호출 순서대로 dict 누적.
    """
    captured: list[dict] = []
    token = _llm_raw_capture.set(captured)
    try:
        yield captured
    finally:
        _llm_raw_capture.reset(token)


# ─────────────────────────────────────────────
#  Client
# ─────────────────────────────────────────────

def _get_client():
    settings = get_settings()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _build_vault_block(vault_context: str) -> str:
    """vault_context 가 있으면 prompt 합류용 block 으로 wrap.

    Phase A B-1 (PI-REVIVE 2026-04-26):
      services.vault_renderer.render_vault_context() 산출물을 LLM prompt 의
      가장 권위 있는 입력으로 합류. 빈 문자열이면 빈 string 반환 → prompt 변화 0.
    """
    s = (vault_context or "").strip()
    if not s:
        return ""
    return (
        "\n══════════════════════════════════════════════\n"
        "## ★ PRIMARY ANALYSIS INPUT — 유저 vault 데이터\n"
        "══════════════════════════════════════════════\n"
        "아래는 유저가 SIGAK 에서 누적한 실 데이터 (Sia 대화 / 추구미 IG·Pinterest 분석 / 자기 표현 원어).\n"
        "**이 데이터가 분석의 PRIMARY 근거입니다.** 좌표 산출 / 추구 방향 / 페르소나 톤 모두 여기서 출발.\n"
        "유저 발화/원어/추구 narrative 를 그대로 활용하세요. 일반론 금지.\n\n"
        f"{s}\n"
        "══════════════════════════════════════════════\n\n"
    )


def _call_llm(system: str, user: str, max_tokens: int = 2048) -> str:
    import time
    settings = get_settings()
    client = _get_client()
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=settings.llm_model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = response.content[0].text
            # 데이터 기업 원칙 — raw 응답 영구 보존 hook.
            # caller 가 with llm_raw_capture() 안에서 호출하면 자동 누적.
            capture = _llm_raw_capture.get()
            if capture is not None:
                capture.append({
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "raw": text,
                    # system / input 의 식별용 앞 200자 (재구성 디버깅).
                    # 풀 프롬프트는 prompt 코드에 있어 재현 가능 — 출력만 영구.
                    "system_excerpt": (system or "")[:200],
                    "input_excerpt": (user or "")[:200],
                    "model": settings.llm_model,
                    "max_tokens": max_tokens,
                    "attempt": attempt + 1,
                })
            return text
        except Exception as e:
            err_type = type(e).__name__
            print(f"[LLM] attempt {attempt+1} failed: {err_type}: {e}")
            if attempt == 0 and ("rate" in str(e).lower() or "overloaded" in str(e).lower() or "529" in str(e)):
                time.sleep(2)
                continue
            raise


# ─────────────────────────────────────────────
#  1. Interview Interpretation
#     "뉴진스 같은데 좀 더 성숙" → aspiration coordinates
# ─────────────────────────────────────────────

def _build_interview_system(gender: str = "female") -> str:
    """유형 앵커 데이터에서 동적으로 인터뷰 해석 프롬프트를 생성한다."""
    type_ref = get_type_reference_prompt(gender)
    return f"""당신은 SIGAK 미감 좌표계의 해석 엔진입니다.
유저의 인터뷰 응답을 분석해서, 미감 좌표계의 3개 축 위에 '추구미 좌표'를 산출합니다.

3축 좌표계:
- shape [-1, +1]: Soft(둥글고 부드러운 골격) ↔ Sharp(날카롭고 선명한 골격)
- volume [-1, +1]: Subtle(작고 섬세한 이목구비) ↔ Bold(크고 존재감 있는 이목구비)
- age [-1, +1]: Fresh(어리고 발랄한 비율) ↔ Mature(성숙하고 정돈된 비율)

SIGAK 인상 유형 레퍼런스:
{type_ref}

규칙:
- 유저가 특정 유형이나 키워드를 언급하면 해당 유형의 좌표를 기준점으로 사용
- 별명도 인식하세요 (예: "첫사랑" = 따뜻한 첫사랑, "시크" = 날카롭고 시크)
- 유저가 특정 인물을 언급하면, 해당 인물의 구조적 특성을 기반으로 가장 유사한 유형 좌표에 매핑
- 수식어("좀 더 성숙", "근데 좀 자연스럽게")는 해당 축에 오프셋 적용
- 불분명한 경우 0에 가까운 중립값 사용
- 반드시 JSON만 출력, 다른 텍스트 금지

출력 형식:
{{"coordinates": {{"shape": 0.1, "volume": 0.3, "age": -0.3}},
  "reference_base": "따뜻한 첫사랑",
  "interpretation": "따뜻한 첫사랑 기반에 성숙도를 높인 방향",
  "confidence": 0.8
}}"""


def interpret_interview(
    interview_data: dict,
    gender: str = "female",
    vault_context: str = "",
) -> dict:
    """
    Parse interview responses into aspiration coordinates.

    interview_data: {
        "desired_image": "뉴진스 같은데 좀 더 성숙한 느낌",
        "reference_celebs": "카리나, 한소희",
        "style_keywords": "시크, 모던, 깔끔",
        "current_concerns": "너무 동안이라 진지해 보이고 싶다",
        ...
    }

    vault_context (Phase A B-1 — PI-REVIVE 2026-04-26):
        services.vault_renderer.render_vault_context() 산출물.
        Sia 대화 + 추구미 이력 + 유저 원어 등 양질의 vault 데이터.
        비어있지 않으면 prompt 가장 권위 있는 입력으로 합류.
        height/weight 등 numeric/factual 은 interview_data 우선.
    """
    # 이미지 키워드: style_image_keywords(신규) 또는 style_keywords(레거시) 사용
    image_kw = interview_data.get('style_image_keywords') or interview_data.get('style_keywords', '없음')

    # Phase A B-1 — vault_context 가 있으면 prompt 권위 입력으로 prepend
    vault_block = _build_vault_block(vault_context)
    has_vault = bool(vault_block)

    if has_vault:
        # vault 가 PRIMARY → interview 는 보조 (대부분 비어있을 가능성)
        user_prompt = f"""유저의 추구미 좌표를 산출해주세요.
{vault_block}
[보조 정보 — 위 vault 데이터가 부족할 때만 활용]
- 추구 이미지: {interview_data.get('desired_image', '없음')}
- 레퍼런스: {interview_data.get('reference_celebs', '없음')}
- 이미지 키워드: {image_kw}
- 현재 고민: {interview_data.get('current_concerns', '없음')}
- 자기 인식: {interview_data.get('self_perception', '없음')}
- 얼굴 고민: {interview_data.get('face_concerns', '없음')}
- 체형: 키 {interview_data.get('height', '미입력')} / 체중 {interview_data.get('weight', '미입력')} / 어깨 {interview_data.get('shoulder_width', '미입력')} / 목 {interview_data.get('neck_length', '미입력')}
- 메이크업: {interview_data.get('makeup_level', '없음')}

분석 가이드:
1. vault 의 유저 발화/원어/추구 narrative 를 PRIMARY 근거로 좌표 산출
2. 추구미 분석 이력의 갭 벡터가 있으면 그 방향으로 aspiration_coords 이동
3. interpretation 필드에 유저 원어를 1-2개 그대로 인용 (예: "유저가 '메시처럼 날렵하고 역동적' 인상을 추구")
4. 보조 정보가 '없음' 이어도 vault 만으로 충분히 산출 가능

JSON으로만 응답해주세요."""
    else:
        # vault 없음 — 옛 path (legacy users / fallback)
        user_prompt = f"""다음 인터뷰 응답에서 추구미 좌표를 산출해주세요.

[추구 이미지]
{interview_data.get('desired_image', '없음')}

[레퍼런스]
{interview_data.get('reference_celebs', '없음')}

[이미지 키워드]
{image_kw}

[현재 고민]
{interview_data.get('current_concerns', '없음')}

[자기 인식]
{interview_data.get('self_perception', '없음')}

[얼굴 고민 영역]
{interview_data.get('face_concerns', '없음')}

[체형]
키: {interview_data.get('height', '미입력')} / 체중: {interview_data.get('weight', '미입력')} / 어깨: {interview_data.get('shoulder_width', '미입력')} / 목: {interview_data.get('neck_length', '미입력')}

[메이크업 레벨]
{interview_data.get('makeup_level', '없음')}

JSON으로만 응답해주세요."""

    system_prompt = _build_interview_system(gender)

    # 진단 로깅 — vault block 합류 여부 + prompt 크기
    print(f"[INTERPRET_PROMPT] has_vault={has_vault} prompt_len={len(user_prompt)} system_len={len(system_prompt)}")

    raw = _call_llm(system_prompt, user_prompt, max_tokens=512)
    print(f"[INTERPRET_RESULT] raw_len={len(raw)} preview={raw[:300]!r}")

    # Parse JSON response
    try:
        # Strip markdown code fence if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: return neutral coordinates
        result = {
            "coordinates": {"shape": 0, "volume": 0, "age": 0},
            "reference_base": "unknown",
            "interpretation": "파싱 실패 — 수동 보정 필요",
            "confidence": 0.0,
        }

    return result


# ─────────────────────────────────────────────
#  2. Report Generation (v2 — Action Spec 기반)
#     Action Spec이 의사결정, Claude는 해설만 담당
# ─────────────────────────────────────────────

REPORT_SYSTEM_V2 = """역할: 당신은 스타일링 해설가입니다.
아래 이미 결정된 추천을 유저 친화적으로 설명하세요.

규칙:
- 추천 항목을 임의로 추가하거나 삭제하지 마세요
- action_tips는 recommended_actions의 순서를 그대로 유지하세요
- 축 점수나 delta 수치를 직접 언급하지 마세요
- "~할 수 있습니다", "~할 수 있어요", "~경향이 있다", "~편이에요" 등 유보적 표현 절대 금지
- 단정형으로 말하세요: "~해보세요", "~줍니다", "~어울려요"
- 부드럽게 하고 싶으면 유보 표현 대신 별도 문장으로 분리하세요
- 같은 내용을 다른 표현으로 반복하지 마세요
- 해요체를 사용하세요
- action_tips 각 항목의 zone 필드는 입력값을 그대로 복사하세요 (번역/의역 금지)

## 금지 표현 목록 (절대 사용 금지)
- 유보: "~경향이 있다", "~할 수 있다", "~편이에요", "~보일 수 있어요"
- 허락: "~해도 돼요", "~괜찮아요", "~나쁘지 않아요"
- 게이밍/캐주얼: "치트키", "꿀팁", "핵꿀", "개꿀", "간지"
- "느낌" 남발: "부드러운 느낌" → "부드러운 무드" 또는 "부드러워요"로 교체
- 한영 동어반복: "소프트한 부드러운", "볼드한 강렬한" → 한 쪽만 사용
- 오탈자: "코등" → "콧등", "코볼" → "콧볼"

## 톤 가이드
- 전문가가 1:1 상담하듯 직접적이고 단정적으로
- "~해보세요"는 OK, "~하면 좋을 것 같아요"는 금지
- 남성(male)에게는 "메이크업" 대신 "그루밍", "스킨케어", "헤어스타일링" 중심
- 여성(female)에게는 "메이크업", "헤어", "스타일링" 자유롭게

각 추천에 대해:
1. 왜 이 영역인지 한 줄 이유
2. 초보자용 적용 팁 1개 (구체적, 실행 가능)
3. 주의할 점 (avoid가 있으면)

## summary 규칙 (필수)
- 반드시 현재 인상과 추구 방향의 차이를 1문장 이상 포함하세요
- 핵심 action 방향을 1문장 이상 포함하세요
- 유저의 퍼스널컬러 타입이 주어지면, "이 방향으로 가려면 컬러는 ___계열을 잡으세요" 식의 톤 가이드를 1문장 포함하세요
- 최소 2문장, 최대 4문장
- 금지: "스타일링을 추천해요" 수준의 일반론만으로 끝내기
- 필수 포함: 매칭 유형명, 추구 방향, 구체적 포인트 1개 이상

반드시 아래 JSON 구조로만 응답하세요. 다른 텍스트를 포함하지 마세요.

{
  "summary": "전체 요약 2~4문장 (현재 인상 + 추구 방향 + 핵심 action)",
  "action_tips": [
    {
      "zone": "영역명 (입력 그대로)",
      "title": "추천 제목",
      "description": "설명 2~3문장",
      "beginner_tip": "초보자 팁 1문장"
    }
  ],
  "avoid_tip": "주의할 점 1~2문장 (없으면 null)",
  "closing": "마무리 한 줄"
}"""


def _get_trend_framing() -> str:
    """트렌드 프레이밍 한 문단. trend_data.py 없으면 빈 문자열."""
    try:
        from pipeline.trend_data import TREND_SEASON, TREND_DIRECTION
        return (
            f"현재 시즌: {TREND_SEASON}. "
            f"{TREND_DIRECTION.get('summary_kr', '')} "
            "트렌드는 참고 맥락이지 절대 기준이 아닙니다. 얼굴 구조 적합성이 항상 우선합니다."
        )
    except ImportError:
        return "트렌드 데이터 없음"


def generate_report(action_spec, user_context: dict) -> str:
    """
    Action Spec 기반 리포트 생성 (v2).
    Claude에게 최소 입력만 전달하고, 해설만 받는다.

    Args:
        action_spec: ActionSpec 인스턴스
        user_context: {"name": str, "face_shape": str, "tier": str, "gender": str}

    Returns:
        Claude raw 응답 문자열 (parse_or_fallback으로 후처리)
    """
    prompt_payload = {
        "user_name": user_context.get("name", ""),
        "face_shape": user_context.get("face_shape", ""),
        "tier": user_context.get("tier", "basic"),
        "matched_type": action_spec.matched_type_label,
        "primary_change_direction": action_spec.primary_gap_axis,
        "recommended_actions": [
            {"순서": a.priority, "영역": a.zone, "방법": a.method, "효과": a.goal}
            for a in action_spec.recommended_actions
        ],
        "avoid_actions": [
            {"영역": a.zone, "이유": a.reason}
            for a in action_spec.avoid_actions
        ],
        "expected_effects": action_spec.expected_effects,
    }

    # summary_context 구성
    aspiration_summary = user_context.get("aspiration_summary", "")
    gap_direction_kr = user_context.get("primary_gap_direction_kr", "")
    top_goals = [a.goal for a in action_spec.recommended_actions[:2]]

    personal_color = user_context.get("personal_color", "")

    # Phase A B-1 (2026-04-26) — vault_context 가 ctx 에 있으면 권위 입력으로 prepend.
    # 시그니처는 그대로 (ctx dict 키 추가 방식). 기존 caller 모두 호환.
    vault_block = _build_vault_block(user_context.get("vault_context", ""))
    has_vault = bool(vault_block)

    user_prompt = f"""{user_context.get('name', '')}님의 스타일링 추천을 설명해주세요.
{vault_block}
{('vault 데이터의 유저 원어/추구 narrative 를 summary 와 closing 에 직접 인용하세요. action_tips 의 description 도 일반론 대신 vault 의 구체 발화 반영.' if has_vault else '')}

[매칭 유형] {prompt_payload['matched_type']}
[주요 변화 방향] {prompt_payload['primary_change_direction']}
[추구미 해석] {aspiration_summary}
[변화 방향 한글] {gap_direction_kr}
[핵심 액션 목표] {', '.join(top_goals)}
[얼굴형] {prompt_payload['face_shape']}
[성별] {user_context.get('gender', 'female')}
[체형] {user_context.get('height', '')} / {user_context.get('weight', '')}
[퍼스널컬러] {personal_color or '미판정'}

[추천 액션]
{json.dumps(prompt_payload['recommended_actions'], ensure_ascii=False, indent=2)}

[주의 사항]
{json.dumps(prompt_payload['avoid_actions'], ensure_ascii=False, indent=2) if prompt_payload['avoid_actions'] else '없음'}

[기대 효과]
{chr(10).join('- ' + e for e in prompt_payload['expected_effects'])}

[시즌 트렌드 참고]
{_get_trend_framing()}

JSON으로만 응답해주세요."""

    print(f"[REPORT_PROMPT] has_vault={has_vault} prompt_len={len(user_prompt)}")
    raw = _call_llm(REPORT_SYSTEM_V2, user_prompt, max_tokens=1500)
    print(f"[REPORT_RESULT] raw_len={len(raw)} preview={raw[:300]!r}")
    return raw


# ─────────────────────────────────────────────
#  2.5. JSON parse fallback
# ─────────────────────────────────────────────

import re


def parse_or_fallback(raw_text: str, action_spec) -> dict:
    """Claude JSON 파싱. 실패 시 deterministic fallback."""
    # 1차: 직접 파싱
    try:
        return json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2차: fenced code block 제거 후 재시도
    try:
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw_text.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    # 3차: deterministic fallback
    return _build_fallback_report(action_spec)


def _build_fallback_report(action_spec) -> dict:
    """Claude 파싱 실패 시 Action Spec으로 직접 리포트 생성"""
    return {
        "summary": " ".join(action_spec.expected_effects[:2]),
        "action_tips": [
            {
                "zone": a.zone,
                "title": a.goal,
                "description": f"{a.zone} 영역에 {a.method}을 적용해보세요.",
                "beginner_tip": "거울을 보면서 소량부터 시작해보세요.",
            }
            for a in action_spec.recommended_actions
        ],
        "avoid_tip": (
            action_spec.avoid_actions[0].reason if action_spec.avoid_actions else None
        ),
        "closing": "한 가지씩 천천히 시도해보세요.",
    }


# ─────────────────────────────────────────────
#  2.9. Legacy generate_report (하위 호환)
# ─────────────────────────────────────────────

def generate_report_legacy(
    user_name: str,
    tier: str,
    face_features: dict,
    current_coords: dict,
    aspiration_coords: dict,
    gap: dict,
    interview_data: dict,
    aspiration_interpretation: dict,
    similar_types: Optional[list[dict]] = None,
    type_comparisons: Optional[list[dict]] = None,
    cluster_result: Optional[dict] = None,
) -> dict:
    """Legacy report generation — Phase 4에서 제거 예정."""
    return {
        "executive_summary": "리포트 생성 방식이 변경되었습니다.",
        "sections": [],
        "action_items": [],
        "similar_types": [],
        "trend_context": "",
    }


# ─────────────────────────────────────────────
#  3. Face Structure Interpretation
#     raw 수치 → 자연어 해석
# ─────────────────────────────────────────────

FACE_INTERPRET_SYSTEM = """당신은 SIGAK의 얼굴 구조 분석 전문가입니다.
InsightFace 랜드마크에서 추출한 수치 데이터를 기반으로,
유저의 얼굴 구조 특징을 자연어로 해석합니다.

톤:
- 판단이 아닌 관찰. "예쁘다/못생겼다"가 아니라 "이러한 특징을 가지고 있다"
- 따뜻하고 전문적인 어조, 해요체
- 전체적인 조화와 특징적인 포인트를 함께 언급
- "~경향이 있다", "~할 수 있다", "~편이에요" 등 유보적 hedging 표현 절대 금지
- 단정적으로 서술하세요: "둥근 인상을 줘요", "입체적인 구조예요"
- 부드럽게 하고 싶으면 유보 표현 대신 별도 부연 문장을 추가하세요

## 해석문 작성 규칙 (필수)
- 해석 문장에 숫자, 소수점, 도(°), 퍼센트(%)를 절대 포함하지 마세요.
- 숫자는 구조화 JSON 필드(value, percentile)에만 남기세요.
- "각도로", "비율로", "돌출도로", "근접도로" 같은 측정 단위+조사 시작 패턴도 금지합니다.
- 해석문은 반드시 의미 중심으로 시작하세요.
- 서술문은 의미와 인상만 설명하세요.

절대 금지 사항:
- 소수점 숫자 (0.613, 0.192 등) 절대 포함하지 마세요
- 퍼센타일 (P45, 상위 30% 등) 절대 포함하지 마세요
- 좌표값 (-0.70, 0.10 등) 절대 포함하지 마세요
- 각도는 자연어로만 표현하세요 ("날카로운 턱선", "부드러운 곡선")
- 영문 zone 이름 (under_eye, brow_arch 등) 대신 한글 (눈 밑, 눈썹 등) 사용하세요
- "느낌" 남발 금지. "부드러운 느낌" → "부드러운 인상" 또는 "부드러워요"
- 한영 동어반복 금지: "소프트한 부드러운" → "부드러운" 하나만
- 오탈자 주의: "코등"(X) → "콧등"(O), "코볼"(X) → "콧볼"(O)
- "치트키", "꿀팁" 등 캐주얼 표현 금지

- 금지 예: "93.7°의 턱선 각도는 날카로운 편으로"
- 허용 예: "턱선이 날카롭고 또렷한 인상을 만들어요"
- 금지 예: "0.719의 광대 돌출도는 상당히 뚜렷한 편으로"
- 허용 예: "광대가 뚜렷해서 입체적이고 개성 있는 인상을 줘요"

[중요] feature 필드는 반드시 아래 목록에서만 선택하세요. 다른 키를 만들지 마세요:
- jaw_angle (턱선)
- eye_tilt (눈꼬리 기울기)
- eye_width_ratio (눈 크기)
- cheekbone_prominence (광대)
- lip_fullness (입술)
- nose_length_ratio (코 길이)
- nose_bridge_height (코 높이)
- face_length_ratio (얼굴 종횡비)
- forehead_ratio (이마 비율)
- brow_arch (눈썹 아치)
- symmetry_score (대칭도)
- golden_ratio_score (황금비)

5~7개 항목을 선택하여 해석하세요.

반드시 아래 JSON 구조로만 출력하세요. 다른 텍스트 금지.

{
  "overall_impression": "전체적인 인상을 2~3문장으로 요약. 숫자 없이 의미만.",
  "feature_interpretations": [
    {
      "feature": "jaw_angle",
      "label": "턱선",
      "interpretation": "자연어 해석 1~2문장. 숫자 없이 인상과 느낌만."
    }
  ],
  "harmony_note": "얼굴 전체 조화에 대한 1문장 코멘트",
  "distinctive_points": ["특징적인 포인트 1~3개 (숫자 없이)"]
}"""


def interpret_face_structure(face_features: dict, vault_context: str = "") -> dict:
    """
    raw 얼굴 수치를 LLM으로 자연어 해석한다.

    Args:
        face_features: face.py에서 추출한 특징 dict
            jaw_angle, eye_tilt, cheekbone_prominence, lip_fullness,
            face_shape, symmetry_score, golden_ratio_score 등
        vault_context (Phase B-2 — PI-REVIVE 2026-04-26):
            services.vault_renderer.render_vault_context() 산출물.
            비어있지 않으면 prompt 에 PRIMARY tone 입력으로 합류 →
            overall_impression / feature_interpretations 가 유저 vault 발화 echo.

    Returns:
        {
            "overall_impression": "...",
            "feature_interpretations": [...],
            "harmony_note": "...",
            "distinctive_points": [...]
        }
    """
    # Phase B-2 — vault_context 가 있으면 prompt 권위 입력으로 prepend
    vault_block = _build_vault_block(vault_context)
    has_vault = bool(vault_block)

    face_data_block = f"""[얼굴 구조 데이터]
- 얼굴형: {face_features.get('face_shape', 'N/A')}
- 턱선 각도: {face_features.get('jaw_angle', 'N/A')}° (낮을수록 날카로움, 높을수록 부드러움)
- 광대 돌출도: {face_features.get('cheekbone_prominence', 'N/A')} (0~1, 높을수록 돌출)
- 눈 크기 비율: {face_features.get('eye_width_ratio', 'N/A')} (눈 너비 / 얼굴 너비)
- 눈 가로세로비: {face_features.get('eye_ratio', 'N/A')} (높을수록 길쭉한 눈)
- 눈꼬리 기울기: {face_features.get('eye_tilt', 'N/A')}° (+ 올라감 / - 처짐)
- 눈 간격: {face_features.get('eye_spacing_ratio', 'N/A')}
- 눈썹 아치: {face_features.get('brow_arch', 'N/A')}
- 코 길이 비율: {face_features.get('nose_length_ratio', 'N/A')}
- 코 높이: {face_features.get('nose_bridge_height', 'N/A')}
- 입술 풍성도: {face_features.get('lip_fullness', 'N/A')} (0~1)
- 얼굴 종횡비: {face_features.get('face_length_ratio', 'N/A')}
- 이마 비율: {face_features.get('forehead_ratio', 'N/A')}
- 인중 비율: {face_features.get('philtrum_ratio', 'N/A')}
- 대칭도: {face_features.get('symmetry_score', 'N/A')} (0~1, 1=완벽)
- 황금비 근접도: {face_features.get('golden_ratio_score', 'N/A')} (0~1, 1=황금비)
- 피부톤: {face_features.get('skin_tone', 'N/A')}
- 피부 밝기: {face_features.get('skin_brightness', 'N/A')} (0~1)"""

    if has_vault:
        # vault 가 PRIMARY tone source. 얼굴 데이터는 객관 사실 baseline.
        user_prompt = f"""유저의 얼굴 구조를 자연어로 해석해주세요.
{vault_block}
{face_data_block}

분석 가이드 (vault 통합):
1. overall_impression (2-3문장):
   - 유저의 vault 발화/추구 narrative 톤을 반영한 인상 묘사
   - 예: 활동적/역동적 vault → "역동적인 분위기와 잘 어우러지는 둥근 얼굴..."
   - 예: 차분/지적 vault → "차분하고 사려깊은 인상의 둥근 얼굴..."
   - 객관 수치 (얼굴형, 대칭도) 는 그대로 반영. tone 만 vault echo.
2. feature_interpretations: 각 feature 인상을 유저 lifestyle/추구 방향과 연결
   - 예: 광대 → "활기를 만드는 입체감" (active vault) vs "지적인 윤곽" (intellectual vault)
3. harmony_note: vault 톤 (페르소나 B 친근체) 으로 1문장
4. distinctive_points: 3-4개. vault narrative 와 연결되는 포인트 우선

규칙 보존:
- 숫자/도/% 절대 X (system prompt 규칙 그대로)
- "~경향이 있다" 등 hedging 금지
- feature 키는 system prompt 목록에서만 선택

JSON으로만 응답해주세요."""
    else:
        # vault 없음 — legacy path (verdict cache wrapper, etc)
        user_prompt = f"""다음 얼굴 구조 수치를 자연어로 해석해주세요.

{face_data_block}

JSON으로만 응답해주세요."""

    print(f"[FACE_INTERP_PROMPT] has_vault={has_vault} prompt_len={len(user_prompt)}")
    raw = _call_llm(FACE_INTERPRET_SYSTEM, user_prompt, max_tokens=1024)
    print(f"[FACE_INTERP_RESULT] raw_len={len(raw)} preview={raw[:300]!r}")

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "overall_impression": "얼굴 구조 해석 중 오류가 발생했습니다.",
            "feature_interpretations": [],
            "harmony_note": "",
            "distinctive_points": [],
        }

    return result


# ─────────────────────────────────────────────
#  4. Type Match Explanation (Phase B-3)
#     WHY THIS TYPE 3-5 bullets + STYLING DIRECTION 2-3 bullets
# ─────────────────────────────────────────────

TYPE_MATCH_SYSTEM = """당신은 SIGAK 유형 매칭 해설 전문가입니다.
유저의 얼굴 구조 좌표와 매칭 유형 + 추구미 좌표 + vault 데이터를 종합해서,
"왜 이 유형인지" + "이 유형 안에서 어떤 방향으로 스타일링할지" 를 설명합니다.

톤:
- 페르소나 B (친근 비서). "~네요" "~군요" 금지. "~예요" "~해요" 단정형.
- 따뜻하고 직접적. hedging ("~경향이 있다", "~할 수 있어요") 절대 X.
- 유저 vault 발화/lifestyle 이 있으면 그걸 reasons 에 자연스럽게 echo.

출력 규칙:
1. reasons (WHY THIS TYPE) — 3-5 bullets, 각 1-2문장
   - 첫 bullet: 얼굴 구조 핵심 특징 + 유형 매칭 이유
   - 둘째 bullet: 좌표 축 (shape/volume/age) 중 가장 강한 신호 설명
   - 셋째 bullet: 추구미 방향 + 매칭 유형 정합 (gap 작으면 "이미 이 방향에 있음" / 크면 "이 유형에서 시작해서 이동")
   - 4-5 bullet (옵션): vault 데이터 있으면 lifestyle/추구 narrative 와 유형 연결
2. styling_tips (STYLING DIRECTION) — 2-3 bullets, 각 1-2문장
   - 매칭 유형 안에서 어떤 변화를 줘야 할지 구체적 (벗어나지 X)
   - 추구 방향에 맞는 zone (메이크업/그루밍 영역) 강조
   - vault 의 lifestyle context 있으면 자연스럽게 통합 (예: 활동적 vault → "운동할 때도 무너지지 않는" 류)

금지:
- 좌표 숫자 (0.5, -0.3 등) 노출 금지
- 일반론 ("자연스럽게", "조화롭게" 류 단독)
- 영문 zone 키 (under_eye, jawline) — 한글로
- 셀럽 이름 (우리 생성물에는 금지, 단 유저 vault 의 echo 는 OK — 예: 유저가 "메시처럼" 발화 → 인용)

JSON으로만 응답:
{
  "reasons": ["bullet 1", "bullet 2", "bullet 3"],
  "styling_tips": ["tip 1", "tip 2"]
}"""


def explain_type_match(
    top_type: dict,
    current_coords: dict,
    aspiration_coords: dict,
    gap: dict,
    gender: str = "female",
    vault_context: str = "",
) -> dict:
    """매칭 유형 설명 (WHY + STYLING) 을 LLM 으로 생성.

    Phase B-3 (PI-REVIVE 2026-04-26):
      기존 build_why_this_type / build_type_styling_tips 의 deterministic
      template fallback 을 LLM 동적 생성으로 대체. vault_context 가 있으면
      유저 lifestyle / 추구 narrative 가 reasons 에 자연 echo.

    Args:
        top_type: similar_types[0] (name_kr, type_id, similarity_pct, axis_delta)
        current_coords: 현재 좌표 dict
        aspiration_coords: 추구미 좌표 dict
        gap: compute_gap 결과 (vector, primary_direction, magnitude)
        gender: female / male
        vault_context: render_vault_context() 산출물 (옵션)

    Returns:
        {"reasons": [str, ...], "styling_tips": [str, ...]}
        파싱 실패 시 빈 list 반환 (caller 가 deterministic fallback).
    """
    type_name = top_type.get("name_kr", "알 수 없음")
    similarity_pct = top_type.get("similarity_pct", 0)
    axis_delta = top_type.get("axis_delta", {})

    vault_block = _build_vault_block(vault_context)
    has_vault = bool(vault_block)

    gap_magnitude = gap.get("magnitude", 0) if gap else 0
    primary_dir = gap.get("primary_direction", "shape") if gap else "shape"
    gap_label = "작음 (이미 이 방향)" if gap_magnitude < 0.2 else (
        "중간 (이 유형 안에서 변화)" if gap_magnitude < 0.5 else "큼 (이 유형 시작점)"
    )

    user_prompt = f"""유저 매칭 유형 + 추구 방향 + (vault 있으면) lifestyle 을 종합해서 reasons + styling_tips 산출.
{vault_block}
[매칭 유형] {type_name}
[유사도] {similarity_pct}%
[성별] {gender}

[얼굴 좌표 축별 차이 (anchor 대비)]
- shape: {axis_delta.get('shape', 0):+.2f} (음수=둥근 쪽, 양수=날카로운 쪽)
- volume: {axis_delta.get('volume', 0):+.2f} (음수=섬세, 양수=뚜렷)
- age: {axis_delta.get('age', 0):+.2f} (음수=어림, 양수=성숙)

[추구 방향]
- 현재 → 추구 갭 magnitude: {gap_magnitude:.2f} ({gap_label})
- 가장 큰 변화 축: {primary_dir}
- 현재 좌표: shape={current_coords.get('shape', 0):+.2f} volume={current_coords.get('volume', 0):+.2f} age={current_coords.get('age', 0):+.2f}
- 추구 좌표: shape={aspiration_coords.get('shape', 0):+.2f} volume={aspiration_coords.get('volume', 0):+.2f} age={aspiration_coords.get('age', 0):+.2f}

분석 가이드:
- {('vault 의 유저 lifestyle/추구 narrative 가 PRIMARY. 좌표는 보조.' if has_vault else 'vault 없음 → 좌표 + 유형 정보로만 설명.')}
- reasons 3-5 bullets, styling_tips 2-3 bullets.
- 좌표 숫자 직접 노출 금지. "또렷한 골격", "차분한 부피" 등 자연어.

JSON으로만 응답해주세요."""

    print(f"[TYPE_MATCH_PROMPT] has_vault={has_vault} type={type_name} sim={similarity_pct}% gap={gap_magnitude:.2f}")

    try:
        raw = _call_llm(TYPE_MATCH_SYSTEM, user_prompt, max_tokens=600)
        print(f"[TYPE_MATCH_RESULT] raw_len={len(raw)} preview={raw[:300]!r}")

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(cleaned)
        reasons = result.get("reasons", []) if isinstance(result, dict) else []
        tips = result.get("styling_tips", []) if isinstance(result, dict) else []
        return {
            "reasons": [r for r in reasons if isinstance(r, str) and r.strip()],
            "styling_tips": [t for t in tips if isinstance(t, str) and t.strip()],
        }
    except (json.JSONDecodeError, Exception) as e:
        print(f"[TYPE_MATCH_FALLBACK] parse/call error: {e}")
        return {"reasons": [], "styling_tips": []}


# ─────────────────────────────────────────────
#  5. Gap Analysis Narration (Phase B-5)
#     하드코딩 template → LLM 자유도. vault tone echo.
# ─────────────────────────────────────────────

GAP_NARRATE_SYSTEM = """당신은 SIGAK 추구미 갭 해설 전문가입니다.
유저의 현재 좌표 + 추구 좌표 + (vault 있으면) lifestyle 데이터를 종합해서
GAP ANALYSIS 섹션의 narrative 를 만듭니다.

톤:
- 페르소나 B (친근 비서). "~네요" "~군요" "~같아요" 금지.
- "~예요" "~해요" 단정형. hedging 절대 X.
- 따뜻하고 직접적. 일반론 ("자연스럽게", "조화롭게") 금지.
- vault 있으면 유저 발화/lifestyle echo 자연스럽게.

출력 구조:
1. gap_summary (1-2 문장):
   - 가장 큰 변화 + 두번째 변화 통합 narrative.
   - 좌표 숫자 노출 X. 자연어로만 ("부드러운 골격으로 이동 + 성숙한 무드 가미").
   - vault 있으면 유저 lifestyle 톤 반영 ("운동할 때의 역동성을 잃지 않으면서 부드러운 골격으로...").

2. axis_narratives (3개, shape/volume/age 각각):
   - difficulty: "큰 변화" / "중간 변화" / "작은 변화" / "거의 일치" 중 1개 (delta 기반)
   - to_label: "X 방향으로" 형식 (예: "부드러운 방향으로", "성숙한 방향으로", "거의 일치")
   - recommendation (1-2 문장): vault tone reflect. 일반론 X.
     예: "골격에서는 윤곽을 둥글게 풀어주는 게 핵심이에요. 운동할 때도 너무 강하게 잡지 X..."

규칙:
- 좌표 숫자 (0.5, -0.3 등) 절대 노출 X
- 영문 axis 키 (shape, volume, age) 노출 X — 한글 라벨 ("골격", "존재감", "무드")
- 단순 반복 금지 — 각 axis 의 narrative 가 다르게.
- 셀럽 이름 우리 생성물 금지. 단 vault 의 echo 는 OK.

JSON으로만 응답:
{
  "gap_summary": "1-2 문장 통합 narrative",
  "axis_narratives": [
    {"axis": "shape", "difficulty": "큰 변화", "to_label": "부드러운 방향으로", "recommendation": "1-2 문장"},
    {"axis": "volume", "difficulty": "거의 일치", "to_label": "거의 일치", "recommendation": "1 문장"},
    {"axis": "age", "difficulty": "중간 변화", "to_label": "성숙한 방향으로", "recommendation": "1-2 문장"}
  ]
}"""


def narrate_gap_analysis(
    current_coords: dict,
    aspiration_coords: dict,
    gap: dict,
    gender: str = "female",
    vault_context: str = "",
) -> dict:
    """GAP 섹션 narrative LLM 생성. 하드코딩 template 대체.

    Phase B-5 (PI-REVIVE 2026-04-26):
      gap_summary + 3축 axis_narratives (difficulty / to_label /
      recommendation) 를 LLM 으로 한번에 생성. vault 톤 echo.

    Returns:
        {"gap_summary": str, "axis_narratives": [{axis, difficulty, to_label, recommendation}]}
        파싱/호출 실패 시 빈 dict (caller deterministic fallback).
    """
    vault_block = _build_vault_block(vault_context)
    has_vault = bool(vault_block)

    gap_vector = gap.get("vector", {}) if gap else {}
    primary_dir = gap.get("primary_direction", "shape") if gap else "shape"
    magnitude = gap.get("magnitude", 0) if gap else 0

    user_prompt = f"""유저의 현재 → 추구 갭 narrative 를 산출해주세요.
{vault_block}
[현재 좌표]
- 골격 (shape): {current_coords.get('shape', 0):+.2f} (음수=둥근/부드러운, 양수=날카로운/또렷한)
- 존재감 (volume): {current_coords.get('volume', 0):+.2f} (음수=섬세/은은한, 양수=뚜렷한/강렬한)
- 무드 (age): {current_coords.get('age', 0):+.2f} (음수=발랄한/어림, 양수=성숙한)

[추구 좌표]
- 골격: {aspiration_coords.get('shape', 0):+.2f}
- 존재감: {aspiration_coords.get('volume', 0):+.2f}
- 무드: {aspiration_coords.get('age', 0):+.2f}

[갭 벡터]
- 가장 큰 변화 축: {primary_dir}
- 전체 magnitude: {magnitude:.2f}
- shape delta: {gap_vector.get('shape', 0):+.2f}
- volume delta: {gap_vector.get('volume', 0):+.2f}
- age delta: {gap_vector.get('age', 0):+.2f}

분석 가이드:
- {('vault 의 lifestyle/추구 narrative 가 톤 PRIMARY' if has_vault else '좌표만 사용')}
- gap_summary 1-2 문장 (전체 변화 방향 통합)
- axis_narratives 3개 (shape/volume/age 각각, 같은 순서로)
- difficulty 기준 (delta 절대값): >0.4 = "큰 변화" / 0.2-0.4 = "중간 변화" / 0.1-0.2 = "작은 변화" / <0.1 = "거의 일치"
- 거의 일치 시 recommendation 은 "지금 방향 그대로 좋아요" 류
- 좌표 숫자 노출 X. 한글 라벨 ("골격", "존재감", "무드") 만.

JSON으로만 응답해주세요."""

    print(f"[GAP_NARRATE_PROMPT] has_vault={has_vault} primary={primary_dir} mag={magnitude:.2f}")

    try:
        raw = _call_llm(GAP_NARRATE_SYSTEM, user_prompt, max_tokens=700)
        print(f"[GAP_NARRATE_RESULT] raw_len={len(raw)} preview={raw[:300]!r}")

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            return {}
        gap_summary = result.get("gap_summary", "")
        axes = result.get("axis_narratives", [])
        if not isinstance(axes, list):
            axes = []
        # 축 결과 sanity (3개 + axis 키 정합)
        cleaned_axes = []
        for a in axes:
            if not isinstance(a, dict):
                continue
            ax_name = a.get("axis", "")
            if ax_name not in ("shape", "volume", "age"):
                continue
            cleaned_axes.append({
                "axis": ax_name,
                "difficulty": str(a.get("difficulty", "")),
                "to_label": str(a.get("to_label", "")),
                "recommendation": str(a.get("recommendation", "")),
            })
        return {
            "gap_summary": str(gap_summary) if gap_summary else "",
            "axis_narratives": cleaned_axes,
        }
    except (json.JSONDecodeError, Exception) as e:
        print(f"[GAP_NARRATE_FALLBACK] parse/call error: {e}")
        return {}
