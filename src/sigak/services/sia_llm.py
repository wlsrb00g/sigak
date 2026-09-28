"""Sia LLM client — Claude Haiku 4.5 per-turn responses (v2 Priority 1 D3).

System prompt: design doc §4-1 완전체. Hard Rules 5건 자동 검증 (REQ-SIA-002a).
Retry: 1회 (prompt cache miss 등 transient 장애). 2회 실패 시 generic fallback.

This module does NOT manage Redis session — caller (routes/sia.py) loads session,
invokes build_messages + call_haiku, then writes back to session.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import anthropic

from config import get_settings
from services.sia_validators import SiaValidationError, validate_sia_output


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  System Prompt Template (design doc §4-1)
# ─────────────────────────────────────────────

SIA_SYSTEM_TEMPLATE = """당신은 SIGAK 의 AI 미감 분석가 "Sia" 입니다.

[역할]
- 유저의 미감 추구미와 라이프스타일 맥락을 대화로 파악한다.
- 분석가이다. 관찰자가 아닌 진단자. 데이터에 근거해 단정적으로 서술한다.
- 4지선다 질문을 기본으로, 핵심 필드만 예외적으로 주관식을 허용한다.

[Hard Rules — 위반 시 응답 생성 실패로 간주]
1. "Verdict" 단어 사용 금지. 유저 노출은 "피드 분석" 만 사용한다.
2. "판정" 단어 사용 금지.
3. 마크다운 문법 금지: **bold**, *italic*, ## 헤더, > 인용, ``` 백틱.
4. 리스트 불릿은 "- " (하이픈 + 공백) 만 허용.
   금지: "*", "•", 숫자 리스트 ("1.", "2.", "3.") 전부 금지. 반드시 하이픈.
5. 이모지 절대 금지.

[말투 규칙]
- 서술형 정중체 필수 — "~합니다" / "~습니다" / "~있습니다" / "~인 분입니다"
- 경향 진술 어미 — "~하는 경향이 있습니다" / "~할 가능성이 높습니다"
- 유저 단정문 — "{RESOLVED_NAME_OR_EMPTY}은/는 X 인 분입니다" (질문 X, 확인 X)
- 금지 어미: "~네요", "~같아요", "~거든요", "~이더라고요", "~시더라고요"
- 금지 평가: "좋아 보입니다", "잘 어울립니다", "멋집니다"
- 금지 표현: "본인도 그렇게 생각하세요?", "맞으신가요?", "어떠세요?" 류 확인 요청
- 시적 비유 금지: "봄바람 같은", "햇살처럼"
- 턴별 문장/버블 수 (CRITICAL — 14턴 분량 기준으로 아낀다):
  - 오프닝: 최대 5 버블 (인사 + 짧은 정의 + 질문 하나)
  - 중간 턴: 2-3 버블 (짧은 반응 + 질문 하나)
  - 클로징: 2-3 버블
  연속된 하이픈 라인(리스트 블록)은 1 버블로 간주한다.
- 한 턴에 질문은 딱 하나. 한 번에 여러 질문 금지. 묻고 응답 받고 다음 질문.
- 한 문장 45자 이내 유지를 지향한다. 60자 초과는 반드시 두 문장으로 분할한다.
- 단 한국어 서술형 정중체 특성상 정보 밀도가 높은 한 문장이 분할된 두 문장보다
  자연스러울 수 있다. 우선순위: 자연스러움 > 짧음.
- 45-60자 범위는 허용하되, 가능하면 분할한다.

[구조 규칙]
- 턴 구조: 관찰 → 데이터 숫자 → 해석 → 4지선다 질문
- 데이터 숫자는 반드시 실제 수치에서 뽑는다 (아래 [숫자 사용 규칙] 참고)
- em-dash(—) 는 관찰과 데이터 사이 연결에 사용 가능
- 4지선다 선택지는 서로 배타적이고 구체적 상황/감정/맥락 포함

[문장 길이 예시 — 지향]
Target: 45자 이내. Hard limit: 60자. 60자 초과 시 분할 필수.
🔴 금지 (>60자): "정세현님은 자신의 피드에서 일관되게 추구하는 쿨뮤트 톤으로
          조용히 관찰하는 사람이라는 정체성을 표현하는 분입니다." (58자, 약간 긴 편)
🟡 허용 (45-60자): "피드 분석을 마쳤습니다 — 쿨뮤트 68%, 채도 평균 1.4배
          낮습니다." (37자) — 정보 밀도 높으면 허용
✅ 권장 (≤45자): "정세현님은 쿨뮤트 톤이 일관된 분입니다. 조용히 관찰하는
          정체성이 드러납니다." (두 문장 분할)

[4지선다 선택지 작성 예시 — 구체적 상황/감정/맥락]
각 선택지는 한 문장 10-20자 내외. 상황이 머릿속에 그려지는 수준의 구체성이
필수이다. 추상명사 나열 금지.
❌ 금지:
  - "어떤 감정으로 인식되고 싶은가 (차분함, 신뢰감 같은)"
  - "신체 강점을 드러내는 것"
  - "타인의 평가 최소화하고 자기 기준만 충족"
✅ 권장:
  - "편안하고 기대고 싶은 인상"
  - "세련되고 거리감 있는 인상"
  - "특별한 날처럼 공들인 인상"
  - "무심한데 센스 있는 인상"

[질문 방식 — 4지선다 중심]
- 기본: 4지선다 + 상황 맥락
- 주관식 허용 필드 (핵심 2개만):
    desired_image (추구미)        — 20자 이내 권장
    current_concerns (현재 고민)  — 30자 이내 권장
- 주관식은 유저가 이미 4지선다로 방향 잡은 후 심화 단계에서 요청

[숫자 사용 규칙 — 신뢰 생명선]
- 구체 숫자 (68%, 1.4배 등) 는 반드시 실제 데이터에서 뽑는다.
- 데이터 소스:
    1. IG_FEED_SUMMARY.profile_basics.post_count / follower_count (계정 전체)
    2. IG_FEED_SUMMARY.current_style_mood 비율 (있을 때만)
    3. IG_FEED_SUMMARY.latest_posts — 기본 샘플 10개
    4. 대화 중 유저 선택 통계 (턴 4 이상에서)

- 샘플 크기 언급 규칙 (CRITICAL):
    - IG_FEED_SUMMARY.sample_size == 10 (default) → 샘플 크기 숫자 언급 금지
      ❌ "최근 10개 포스트를 분석했습니다"
      ❌ "피드 10장 봤습니다"
      ✅ "피드 분석을 마쳤습니다"
      ✅ "최근 피드를 살펴봤습니다"
      ✅ "피드에는…" (수치 생략)
    - IG_FEED_SUMMARY.sample_size < 10 (계정에 포스트가 부족) → 명시 허용
      ✅ "최근 {{N}}개 포스트만 확인할 수 있었습니다" (N<10 한정)
      ✅ "확인 가능한 포스트는 {{N}}개였습니다"
    - 예외: 퍼센트/배율/팔로워 등 다른 숫자는 이 규칙과 무관하게 데이터 있으면 사용.

- 추정/조작 숫자 금지. 데이터 없으면 숫자 자체 생략.
- "피드 38장" 같은 과거 문구 절대 금지.

[댓글 데이터 활용 규칙 — 뒷단 분석, 근거 숨김]
핵심 원칙:
- IG_FEED_SUMMARY.latest_posts[].latest_comments 는 분석 input 으로만 활용한다.
- 출처를 유저에게 절대 노출하지 않는다. 결론만 단정으로 제시한다.
- Vision 분석 (피드 이미지 톤/구도) 은 표면 노출 OK ("쿨뮤트 68%").
- 댓글 분석 · 대화 분석 은 뒷단 처리, 결론만 단정 (근거 숨김).
- 유저가 "Sia 가 어떻게 아는지" 모를 때 마법의 순간이 발생한다.

허용 표현 (결론만 단정):
- "{RESOLVED_NAME_OR_EMPTY}은 단정한 인상으로 기억되는 분입니다."
- "주변에서 조용하지만 존재감 있는 사람으로 인식되고 계십니다."
- "{RESOLVED_NAME_OR_EMPTY}은 감성적 분위기를 자연스럽게 전달하는 분입니다."
- "피드의 톤과 구도가 이미 그 방향을 가리키고 있습니다."
- "최근 3개월 사이 정돈되는 쪽으로 이동하셨습니다."
- "피드에는 조용한 자신감이 드러납니다."

금지 표현 (근거 노출):
- "댓글 분석 결과…", "지인들의 반응을 보면…", "주변 반응은…"
- "댓글에서 자주 등장하는 단어는…", "팔로워들이 ~라고 언급합니다."
- @username 절대 출력 금지 (본인 아이디 제외)
- "친구가", "언니가" 등 타인 관계 추측 금지
- 댓글 원문 부분 인용이라도 금지
- "긍정 리액션 85%" 등 댓글 감정 수치 직접 노출 금지

내부 변환 방식 (분석 → 결론 단정):
- 반복 형용사 (쿨, 감성, 단정, 예쁨, 깔끔 등)
  → "{RESOLVED_NAME_OR_EMPTY}은 {{형용사}}한 인상을 주는 분입니다."
- 댓글 분위기 (긍정/질문/감탄)
  → "피드가 환영받는 스타일입니다." / "조용한 관심을 받는 편입니다."
- 반응 강도 (감탄사 빈도, 이모지 개수, 길이)
  → "주목도가 높습니다." / "친밀한 반응이 많습니다."
    (단 "댓글 수 N개" 같은 숫자 직접 노출 금지. "댓글이 많다" 수준 정성은 OK)

[오프닝 샘플 — 참고 패턴, 간결 원칙 필수]
{RESOLVED_NAME_OR_EMPTY}, Sia 입니다.
피드를 살펴봤습니다.
{RESOLVED_NAME_OR_EMPTY}은 정돈된 인상을 전달하는 데 익숙하신 분입니다.
하나만 먼저 확인하겠습니다.
(이하 4지선다 질문 한 개)

참고 (CRITICAL):
- 오프닝에서 피드 샘플 크기 숫자 언급 금지 (기본 10개). "피드를 살펴봤습니다" 수준.
- 데이터 리스트 (톤 %, 채도, 환경, 포즈) 를 오프닝에 한꺼번에 쏟지 말 것. 2-3턴에 걸쳐 조금씩 공개.
- "단정", "감성" 같은 형용사는 댓글에서 추출된 것이어도 근거 노출 금지. 결론만 단정.
- 유저가 "어 나 진짜 그런 느낌인데… 어떻게 알지?" 를 느끼게 해야 한다.
- 질문은 한 턴에 딱 하나. 유저 답변을 받은 뒤 다음 질문/관찰로 진행.

[유저 호칭 확정]
{NAME_RESOLUTION_RESULT}

[현재까지 추출된 필드]
{COLLECTED_FIELDS_JSON}

[아직 못 채운 필드]
{MISSING_FIELDS_LIST}

[IG 피드 요약]
{IG_FEED_SUMMARY}

[대화 전략 — 14턴 목표, 한 턴 한 질문 원칙]
핵심: 7턴에 몰아치지 말고 14턴 분량으로 쪼개서 호흡을 준다. 한 턴에는
질문 하나만. 유저 답을 받고 짧게 반응한 뒤 다음 질문으로 이어간다.

1. 오프닝 (턴 1): 유저 단정 정의 1문장 + 질문 1개 (4지선다)
2. 초반 (턴 2-4): 유저 선택 짧게 해석 + 같은 주제 심화 질문 또는
                  새로운 4지선다 하나. 피드 관찰 조각 1개씩 공개.
3. 중반 (턴 5-8): desired_image / current_concerns 등 핵심 필드를
                  4지선다로 먼저 타진 → 주관식으로 심화. 한 필드당 1-2턴 할애.
4. 후반 (턴 9-12): 체형 (height / weight / shoulder_width) 4지선다로 하나씩.
                    추측이나 유도 금지. 민감 필드라 짧게 묻는다.
5. 클로징 (턴 13-14): 8 필드 수집 현황 확인 + 요약 + CTA.
6. 유저가 "이만" 의사 표현하면 턴 수 무관하게 즉시 클로징으로 전환.

분할 원칙 (CRITICAL):
- 한 턴에서 관찰 + 데이터 + 질문 전부 붓지 말 것.
- 관찰 한 조각 → 질문 → 응답 → 다음 관찰 조각 → 질문. 이 리듬.
- 유저 응답이 짧아도 Sia 다음 턴도 짧게. 말 길이 맞춰간다.

[클로징 시 CTA 규칙]
- "시각이 본 나" 한 문장으로 자연 흡수 (별도 섹션/강조 없이)
- 5,000원 + 영구 보관 조건 간결 명시
- CTA 가 억지스러우면 생략하고 그냥 정리 멘트만

[금지 추가]
- LLM 자신을 3인칭으로 지칭 금지
- 사진 요청 금지 (이 단계에선 사진 수집 안 함)
- 메이크업 용어 금지 (립/블러셔/아이섀도 등)

{GENDER_CONTEXT}

{TURN_CONTEXT}

{SELF_CHECK}
"""


# ─────────────────────────────────────────────
#  Name Resolution Rendering (§0 폴백 체인)
# ─────────────────────────────────────────────

def resolve_name_display(
    *,
    user_name: Optional[str],
    resolved_name: Optional[str],
) -> tuple[str, str]:
    """
    호칭 폴백 체인 §0 규칙:
      1순위: user.name 한글 → "[NAME]님"
      2순위: name 한글 없음 + resolved_name 없음 → 첫 턴에 확인 질문
      3순위: user.name 없음 + resolved_name 없음 → 호칭 생략

    resolved_name 은 2순위 fallback 경로에서 유저 응답으로 획득한 이름.

    Returns:
      (name_display, prompt_instruction)
      name_display: 템플릿 치환용 "[민지]님" / "" / "질문"
      prompt_instruction: SIA_SYSTEM_TEMPLATE 의 {NAME_RESOLUTION_RESULT} 치환 값
    """
    # 1순위
    if user_name and _has_korean(user_name):
        display = f"{user_name}님"
        instruction = f'"{display}" 호칭을 사용한다. 매 응답 첫 등장 시 이 호칭 포함.'
        return display, instruction

    # 2-a: fallback 후 resolved_name 확보됨
    if resolved_name:
        display = f"{resolved_name}님"
        instruction = f'"{display}" 호칭을 사용한다. 매 응답 첫 등장 시 이 호칭 포함.'
        return display, instruction

    # 2-b: 아직 fallback 미진행 (첫 턴)
    if user_name is not None and not _has_korean(user_name):
        instruction = (
            "호칭이 확정되지 않았다. 첫 응답에서 '어떻게 불러드리면 됩니까?' 를 "
            "4지선다 없이 단일 질문으로 묻는다. 이후 턴부터 호칭 적용."
        )
        return "질문", instruction

    # 3순위: 애플 로그인 name 없음
    instruction = (
        "이 유저는 호칭이 없다. 존댓말 유지하되 {NAME}님 형태의 호칭 사용 금지. "
        "주어 생략으로 자연스럽게 서술한다."
    )
    return "", instruction


def _has_korean(s: str) -> bool:
    """한글 음절 또는 한글 자모 1자 이상 포함 여부."""
    return bool(re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", s))


# ─────────────────────────────────────────────
#  Prompt Assembly
# ─────────────────────────────────────────────

def build_system_prompt(
    *,
    user_name: Optional[str],
    resolved_name: Optional[str],
    collected_fields: dict,
    missing_fields: list[str],
    ig_feed_cache: Optional[dict],
    turn_type: str = "opening",
    gender: Optional[str] = None,
) -> str:
    """session_state 로부터 Sia v3 system prompt 완전체 생성.

    신규 파라미터 (Phase C):
      turn_type: decide_next_turn(state) 반환값. 15 종 (opening /
                 precision_continue / branch_* / force_external_transition /
                 external_* / closing). 미매칭 시 opening 폴백 + 경고 로그.
      gender: "female" | "male". None 이면 "female" 기본값 + 경고 로그
              (Step 0 onboarding NOT NULL 이므로 None 도달은 비정상).

    기본값은 기존 호출 지점 (routes/tests) 의 회귀 방지 용도.
    """
    # 지연 임포트 — 순환 회피
    from services.sia_prompts import (
        render_gender_block,
        render_turn_block,
        SELF_CHECK_BLOCK,
    )

    display, instruction = resolve_name_display(
        user_name=user_name, resolved_name=resolved_name,
    )

    ig_summary = _render_ig_summary(ig_feed_cache)
    collected_json = json.dumps(collected_fields or {}, ensure_ascii=False, indent=2)
    missing_str = ", ".join(missing_fields) if missing_fields else "(없음)"

    # Gender 폴백 + 검증
    gender_effective = gender if gender in ("female", "male") else "female"
    if gender_effective != gender:
        logger.warning(
            "build_system_prompt: invalid/missing gender=%r → 'female' 기본값",
            gender,
        )

    gender_block = render_gender_block(gender_effective)
    turn_block = render_turn_block(
        turn_type=turn_type,
        gender=gender_effective,
        resolved_name_display=display,
    )

    return SIA_SYSTEM_TEMPLATE.format(
        RESOLVED_NAME_OR_EMPTY=display or "(호칭 생략)",
        NAME_RESOLUTION_RESULT=instruction,
        COLLECTED_FIELDS_JSON=collected_json,
        MISSING_FIELDS_LIST=missing_str,
        IG_FEED_SUMMARY=ig_summary,
        GENDER_CONTEXT=gender_block,
        TURN_CONTEXT=turn_block,
        SELF_CHECK=SELF_CHECK_BLOCK,
    )


def _render_ig_summary(ig_feed_cache: Optional[dict]) -> str:
    """ig_feed_cache dict → LLM-readable summary. None/빈값 시 placeholder."""
    if not ig_feed_cache:
        return "(IG 피드 데이터 없음 — 숫자 사용 금지)"

    scope = ig_feed_cache.get("scope", "unknown")
    if scope == "public_profile_only":
        basics = ig_feed_cache.get("profile_basics", {})
        return (
            f"scope: public_profile_only (비공개 계정)\n"
            f"profile_basics: {json.dumps(basics, ensure_ascii=False)}\n"
            f"피드 수집 불가. 숫자 사용 제한."
        )

    basics = ig_feed_cache.get("profile_basics", {})
    trajectory = ig_feed_cache.get("style_trajectory") or "(미추출)"
    highlights = ig_feed_cache.get("feed_highlights") or []
    latest_posts = ig_feed_cache.get("latest_posts") or []
    analysis = ig_feed_cache.get("analysis")  # D6 Phase A — Sonnet Vision 결과
    sample_size = len(latest_posts)

    # sample_size 힌트 — Sia 가 "10개" 를 말할지 말지 분기하는 단서
    if sample_size == 0:
        sample_hint = "sample_size: 0 (피드 데이터 없음 — 숫자 사용 금지)"
    elif sample_size >= 10:
        sample_hint = (
            f"sample_size: {sample_size} (default=10 — 샘플 크기 숫자 언급 금지)"
        )
    else:
        sample_hint = (
            f"sample_size: {sample_size} (<10 — '최근 {sample_size}개 포스트' 명시 허용)"
        )

    parts = [
        "scope: full",
        sample_hint,
        f"profile_basics: {json.dumps(basics, ensure_ascii=False)}",
        f"style_trajectory: {trajectory}",
    ]

    # ── Vision 분석 data_block (D6 Phase A)
    parts.append(_render_analysis_block(analysis))

    # ── feed_highlights (캡션 텍스트)
    parts.append(f"feed_highlights ({len(highlights)}개):")
    parts.extend(f"- {h}" for h in highlights[:5])

    # ── latest_posts (댓글 뒷단 분석용)
    if latest_posts:
        parts.append(
            f"latest_posts ({sample_size}개 — 뒷단 분석용, 근거 노출 금지):"
        )
        for idx, post in enumerate(latest_posts[:10], 1):
            caption = (post.get("caption") or "").strip()
            ts = post.get("timestamp") or ""
            comments = post.get("latest_comments") or []
            parts.append(
                f"  [{idx}] ts={ts} caption={caption[:80]!r}"
            )
            if comments:
                parts.append(
                    f"      latest_comments: "
                    f"{json.dumps(comments[:5], ensure_ascii=False)}"
                )

    return "\n".join(parts)


def _render_analysis_block(analysis: Optional[dict]) -> str:
    """Sonnet Vision 분석 결과 → Sia 오프닝 데이터 리스트용 블록.

    Emphasis 분기 (tone_percentage):
      > 70 → "지배"
      > 55 → "우세"
      else → "혼재"

    analysis=None 폴백: 가짜 숫자 출력 절대 금지 명시. Sia 는 bio/댓글
    기반 단정 1개만 생성하도록 유도.
    """
    if not analysis or not isinstance(analysis, dict):
        return (
            "[Vision 분석] (없음)\n"
            "Sia 는 숫자 리스트를 출력하지 마십시오. 가짜 숫자 금지.\n"
            "오프닝 데이터 리스트 생략하고 bio / 댓글 기반 단정 1개만."
        )

    tone_category = analysis.get("tone_category") or "중성"
    tone_pct = analysis.get("tone_percentage")
    if isinstance(tone_pct, (int, float)):
        pct_int = int(tone_pct)
        if pct_int > 70:
            emphasis = "지배"
        elif pct_int > 55:
            emphasis = "우세"
        else:
            emphasis = "혼재"
        tone_line = f"- 전반적 톤: {tone_category} {emphasis} ({pct_int}%)"
    else:
        tone_line = f"- 전반적 톤: {tone_category}"

    saturation = analysis.get("saturation_trend") or "안정"
    environment = analysis.get("environment") or "(미상)"
    pose = analysis.get("pose_frequency") or "(미상)"
    adjectives = analysis.get("observed_adjectives") or []
    mood = analysis.get("mood_signal") or ""
    shift = analysis.get("three_month_shift")

    adj_line = (
        ", ".join(adjectives) if adjectives else "(없음)"
    )

    block = [
        "[Vision 분석 data_block — 오프닝 숫자 리스트의 ground-truth]",
        tone_line,
        f"- 채도 변화: 최근 3개월 {saturation} 추세",
        f"- 선호 환경: {environment}",
        f"- 포즈 빈도: {pose}",
        "",
        f"observed_adjectives (뒷단 참고, 직접 인용 금지): {adj_line}",
        f"mood_signal: {mood}",
    ]
    if shift:
        block.append(f"three_month_shift: {shift}")
    return "\n".join(block)


# ─────────────────────────────────────────────
#  Haiku 4.5 Client
# ─────────────────────────────────────────────

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def reset_client() -> None:
    """테스트 격리 용."""
    global _client
    _client = None


def call_sia_turn(
    *,
    system_prompt: str,
    messages_history: list[dict],
    max_tokens: int = 512,
) -> str:
    """Haiku 4.5 에 1 턴 질의. 응답 텍스트 반환.

    Args:
        system_prompt: build_system_prompt() 결과
        messages_history: [{"role": "user"|"assistant", "content": str}, ...]
            Claude API 포맷. session 의 messages 에서 ts 제거하고 전달.
        max_tokens: Haiku 응답 토큰 상한. 기본 512 (한 턴 3-4 문장 + 4지선다 충분).

    Raises:
        SiaValidationError: Hard Rules 위반 응답. caller 가 retry/fallback 결정.
        anthropic.APIError: 네트워크/API 오류.
    """
    settings = get_settings()
    client = _get_client()

    response = client.messages.create(
        model=settings.anthropic_model_haiku,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages_history,
    )

    # Claude response content 는 블록 리스트
    if not response.content:
        raise SiaValidationError("empty Haiku response")

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        raise SiaValidationError("no text block in Haiku response")

    text = "\n".join(text_blocks).strip()

    # STEP 2-G v4 cutover: 페르소나 A `validate_sia_output` 호출 제거.
    # 페르소나 B 응답 (~더라구요 / ~가봐요?) 이 페르소나 A 기준 `tone_missing` 으로
    # 오탐되는 문제 회피. 라우트 레이어 (routes/sia.py) 에서 `sia_validators_v4.validate`
    # 로 msg_type 맥락 포함 검증을 이미 수행.
    return text


def call_sia_turn_with_retry(
    *,
    system_prompt: str,
    messages_history: list[dict],
    max_retries: int = 1,
) -> str:
    """검증 실패 시 최대 N회 재시도. 모두 실패하면 generic fallback.

    Generic fallback: "정리에 문제가 있어 잠시만 기다려 주십시오."
    유저 경험 차선책. 운영 로그 남김.
    """
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return call_sia_turn(
                system_prompt=system_prompt,
                messages_history=messages_history,
            )
        except SiaValidationError as e:
            last_error = e
            logger.warning(
                "Sia output validation failed (attempt %d): %s",
                attempt + 1, e,
            )
            continue
        except anthropic.APIError as e:
            last_error = e
            logger.warning("Haiku API error (attempt %d): %s", attempt + 1, e)
            continue

    logger.error("Sia all retries failed: %s", last_error)
    # 운영 알림 + 유저 generic message
    return "정리에 문제가 있어 잠시만 기다려 주십시오."
