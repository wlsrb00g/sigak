"""Sia 다음 메시지 결정 트리 (14 타입 + 세션 #6/7 완결판).

SPEC 출처: .moai/specs/SPEC-SIA/
  - 세션 #4 v2 §6 (M1 결합 출력)
  - 세션 #6 v2 §8.1 (decide_next_message 기본 우선순위)
  - 세션 #7 §1-7 (EMPATHY 결합 / C6 / C7 / A-13 재배치)

우선순위 (최상위 → 하위, 세션 #7 반영 재배치):
  0. M1 결합 출력 (len(a_turns)==0)
  1. A-11 과몰입 (최우선 트리거)
  2. CHECK_IN 직후 → RE_ENTRY (+ exit_confirmed 판정)
     (세션 #6 v2 §3 주석 "이탈 선택이든 재참여든 모두 RE_ENTRY" 에 맞춰 A-10 보다 앞)
  3. A-10 직접 이탈 신호 → CHECK_IN
  4. A-9 단답 스트릭 ≥ 3 → CHECK_IN 이양
  5. EMPATHY 결합 출력 — emotion / tt / self_disclosure 2연속 (세션 #7 §1.3)
     secondary 는 eval_request + rich 판정 으로 C6 / REAFFIRM / RECOGNITION / PROBE 분기
  6. C6 평가 요청 — 감정 없는 독립 eval_request (자기개시 풍부 → C6 / 막막함 → REAFFIRM)
  7. C7 일반화 회피
  8. A-3 other 트리거 (explain / meta / evidence)
  9. A-9 단답 스트릭 1-2 분기 (B 경로 / D 경로)
 10. A-2 비율 강제 (수집 3연속 / RECOGNITION 하한 / DIAGNOSIS 하한)
 11. Phase 기본 (concede / defensive / obs<3 순환 / DIAGNOSIS 후 SOFT_WALKBACK)

결정적 로직. Haiku/LLM 호출 없음. state + 정규식만으로 결정.

Public API:
  decide(state) -> Composition           — 전수 메타 포함
  decide_next_message(state) -> MsgType  — primary_type 만 (기존 호출부 호환)
  update_state_from_user_turn(state, text) — 유저 턴 append 후 호출

Detectors (세션 #6/7 패턴):
  is_trivial / detect_exit_signal / detect_overattachment
  detect_eval_request / detect_generalization
  detect_user_disclaimer / detect_self_pr
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from schemas.sia_state import (
    DIAGNOSIS_MIN_RATIO,
    ConversationState,
    MsgType,
    OverattachmentSeverity,
    RangeMode,
    UserMessageFlags,
)


# ─────────────────────────────────────────────
#  A-9 단답 (세션 #6 v2 §2.1)
# ─────────────────────────────────────────────

TRIVIAL_EXACT: frozenset[str] = frozenset({
    "네", "넹", "넵", "응", "ㅇㅇ", "ㅎㅎ", "ㅇㅋ", "아뇨", "아니요",
})

# "잘 모르겠어요" 포함 — 세션 #6 v2 §10.2 = (a) 확정
AMBIGUOUS_SHORT: frozenset[str] = frozenset({
    "맞아요", "그쵸", "글쎄요", "잘 모르겠어요", "그냥요",
})

_CONTENT_WORD_RE = re.compile(r"[가-힣a-zA-Z]{2,}|\d")


def is_trivial(text: str) -> bool:
    """A-9 단답 판정 — 정보 bit 없는 회피성 응답."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped in TRIVIAL_EXACT:
        return True
    if stripped in AMBIGUOUS_SHORT:
        return True
    # 3자 이하 + 한글/영문 2자+ 또는 숫자 없음 → 무의미 단답
    if len(stripped) <= 3 and not _CONTENT_WORD_RE.search(stripped):
        return True
    return False


# ─────────────────────────────────────────────
#  A-10 직접 이탈 (세션 #6 v2 §3.1)
# ─────────────────────────────────────────────

_EXIT_PATTERNS = [
    re.compile(r"나중에"),
    re.compile(r"다음에"),
    re.compile(r"이따가"),
    re.compile(r"지금\s*바빠"),
    re.compile(r"그만"),
    re.compile(r"됐어요"),
    re.compile(r"안\s*할래"),
]


def detect_exit_signal(text: str) -> bool:
    return any(p.search(text or "") for p in _EXIT_PATTERNS)


# ─────────────────────────────────────────────
#  A-11 과몰입 (세션 #6 v2 §4.1)
# ─────────────────────────────────────────────

_OVERATTACH_MILD = [
    re.compile(r"진짜\s*(나|저)\s*잘\s*아"),
    re.compile(r"AI\s*인데"),
    re.compile(r"신기해요"),
    re.compile(r"사람\s*같"),
]
_OVERATTACH_SEVERE = [
    re.compile(r"들어줄\s*사람\s*없"),
    re.compile(r"계속\s*얘기하고\s*싶"),
    re.compile(r"상담받는\s*거\s*같"),
    re.compile(r"친구보다"),
]


def detect_overattachment(text: str) -> tuple[bool, OverattachmentSeverity]:
    """A-11 과몰입 감지. severe 우선 (고립/의존 신호가 강함)."""
    t = text or ""
    for p in _OVERATTACH_SEVERE:
        if p.search(t):
            return True, "severe"
    for p in _OVERATTACH_MILD:
        if p.search(t):
            return True, "mild"
    return False, ""


# ─────────────────────────────────────────────
#  C6 평가 요청 (세션 #7 §2.2)
# ─────────────────────────────────────────────

_EVAL_REQUEST_PATTERNS = [
    re.compile(r"어떻게\s*보(여|이)"),
    re.compile(r"괜찮(나|아|을까)요"),
    re.compile(r"(과해|부족해|이상해|예뻐|매력|어색해)\s*보(여|이)"),
    re.compile(r"있긴\s*한"),
    re.compile(r"제가\s*어때"),
    re.compile(r"평가"),
    re.compile(r"솔직히\s*말"),
    re.compile(r"내(는|가)\s*뭔"),
]


def detect_eval_request(text: str) -> bool:
    return any(p.search(text or "") for p in _EVAL_REQUEST_PATTERNS)


# ─────────────────────────────────────────────
#  C7 일반화 회피 (세션 #7 §3.3)
# ─────────────────────────────────────────────

_GENERALIZATION_PATTERNS = [
    re.compile(r"다들\s*(그|이|저)"),
    re.compile(r"보통\s*(다|그)"),
    re.compile(r"누구나"),
    re.compile(r"원래\s*(다|그)"),
    re.compile(r"남들도"),
    re.compile(r"흔한\s*거"),
    re.compile(r"평범(해|한|하|함)"),
]


def detect_generalization(text: str) -> bool:
    return any(p.search(text or "") for p in _GENERALIZATION_PATTERNS)


# ─────────────────────────────────────────────
#  A-16 유저 무지 명시 (세션 #7 §7.4)
# ─────────────────────────────────────────────

_DISCLAIMER_PATTERNS = [
    re.compile(r"잘\s*몰라"),
    re.compile(r"(딱히|별로)\s*신경.*안"),
    re.compile(r"(딱히|별로)\s*관심.*없"),
    re.compile(r"생각\s*안\s*해\s*봤"),
    re.compile(r"(잘|딱히)\s*모르겠"),
]


def detect_user_disclaimer(text: str) -> bool:
    """유저가 무지/무관심 명시 → N턴 동안 해당 영역 의식/의도 질문 금지."""
    return any(p.search(text or "") for p in _DISCLAIMER_PATTERNS)


# ─────────────────────────────────────────────
#  A-13 자기 PR (세션 #7 §5.2)
# ─────────────────────────────────────────────

_SELF_PR_PATTERNS = [
    re.compile(r"누가\s*봐도"),
    re.compile(r"저는\s*좋(아|은)\s*것\s*같"),
    re.compile(r"자신\s*(있|있어)"),
    re.compile(r"(근육|몸매|외모|얼굴)\s*(좋|잘|괜찮)"),
    re.compile(r"딱\s*보면"),
]


def detect_self_pr(text: str) -> bool:
    """유저가 외모/능력/성취 자체를 직접 호명 → A-13 prefix 허용 조건."""
    return any(p.search(text or "") for p in _SELF_PR_PATTERNS)


# ─────────────────────────────────────────────
#  Pre-decide state updater (라우터에서 호출)
# ─────────────────────────────────────────────

DISCLAIMER_MEMORY_TURNS = 2    # 세션 #7 §7.4 "다음 N턴 금지"
SELF_PR_PREFIX_MAX = 2         # 세션 #7 §5.2 "세션당 1-2회 한정"


def update_state_from_user_turn(
    state: ConversationState, user_text: str,
) -> None:
    """유저 턴 append 직후 + decide 전 라우터가 호출.

    상태 변경:
      - A-9 trivial_streak: is_trivial() 이면 +1, 아니면 0 리셋
      - A-16 user_disclaimer_memory:
          * detect_user_disclaimer 히트 시 "recent" key 를 N (DISCLAIMER_MEMORY_TURNS) 로 set
          * 히트 없으면 기존 메모리 tick (각 키 -1, 0 도달 시 삭제)

    Note: self_pr_prefix_used 는 prefix 실제 적용 시점 (routes/prompts) 에서 증가.
    여기서는 detect 만 가능.
    """
    if is_trivial(user_text):
        state.trivial_streak += 1
    else:
        state.trivial_streak = 0

    if detect_user_disclaimer(user_text):
        state.user_disclaimer_memory["recent"] = DISCLAIMER_MEMORY_TURNS
    else:
        decayed: dict[str, int] = {}
        for topic, remaining in state.user_disclaimer_memory.items():
            if remaining > 1:
                decayed[topic] = remaining - 1
        state.user_disclaimer_memory = decayed


# ─────────────────────────────────────────────
#  Composition — decide 결과 메타
# ─────────────────────────────────────────────

@dataclass
class Composition:
    """decide() 결과. Haiku 프롬프트 조립 + 하드코딩 렌더링에 직접 사용.

    primary_type 외 메타 플래그들은 선택적. 대부분의 케이스에서 secondary_type /
    confrontation_block 등은 None 으로 남아있고 primary_type 만 의미 있다.
    """
    primary_type: MsgType
    secondary_type: Optional[MsgType] = None   # 결합 출력 둘째 문장 타입
    tertiary_type: Optional[MsgType] = None    # 3단 결합 셋째 문장 타입
    confrontation_block: Optional[str] = None  # CONFRONTATION 블록 (C1~C7)
    range_mode: RangeMode = "limit"            # RANGE_DISCLOSURE 모드
    exit_confirmed: bool = False               # RE_ENTRY V5 종결 플래그
    apply_self_pr_prefix: bool = False         # A-13 prefix 적용 플래그

    @property
    def is_combined(self) -> bool:
        """결합 출력 여부 — secondary_type 이 있으면 True."""
        return self.secondary_type is not None


# ─────────────────────────────────────────────
#  Core decision
# ─────────────────────────────────────────────

def decide(state: ConversationState) -> Composition:
    """14 타입 체계 의사결정. 결과 Composition 반환.

    §docstring 우선순위 0-10 순회 후 첫 매칭 반환.
    """
    a_turns = state.assistant_turns()

    # 0. M1 결합 출력 — 오프닝 + 첫 관찰 한 메시지
    if not a_turns:
        return Composition(
            primary_type=MsgType.OPENING_DECLARATION,
            secondary_type=MsgType.OBSERVATION,
        )

    last_user = state.last_user()
    if last_user is None:
        # 어시스턴트 턴만 있고 유저 턴 없음 — 일반 OBSERVATION
        return Composition(primary_type=MsgType.OBSERVATION)

    last_user_text = last_user.text
    flags = last_user.flags
    last_assistant = a_turns[-1]

    # A-13 self-PR prefix 후보 플래그 (세션당 1-2회 한정)
    apply_self_pr_prefix = (
        detect_self_pr(last_user_text)
        and state.self_pr_prefix_used < SELF_PR_PREFIX_MAX
    )

    # 1. A-11 과몰입 (최우선)
    overattach, severity = detect_overattachment(last_user_text)
    if overattach:
        state.overattachment_severity = severity
        return Composition(
            primary_type=MsgType.RANGE_DISCLOSURE,
            range_mode="limit",
        )

    # 2. CHECK_IN 직후 → RE_ENTRY (A-10 보다 앞 — 이탈 의사 무관하게 RE_ENTRY)
    if last_assistant.msg_type == MsgType.CHECK_IN:
        exit_confirmed = detect_exit_signal(last_user_text)
        return Composition(
            primary_type=MsgType.RE_ENTRY,
            exit_confirmed=exit_confirmed,
        )

    # 3. A-10 직접 이탈 → CHECK_IN
    if detect_exit_signal(last_user_text):
        return Composition(primary_type=MsgType.CHECK_IN)

    # 4. A-9 단답 스트릭 ≥ 3 → CHECK_IN 이양
    if state.trivial_streak >= 3:
        return Composition(primary_type=MsgType.CHECK_IN)

    # 5. EMPATHY 결합 출력 — emotion / tt / self_disclosure 2연속 (세션 #7 §1.3)
    #    emotion 트리거를 C6/C7 보다 앞에 둬서 서연 M5 형 ("EMPATHY + C6") 경로 확보.
    if (
        flags.has_emotion_word
        or flags.has_tt
        or _consecutive_self_disclosure(state, n=2)
    ):
        return _empathy_combined(state, flags)

    # 6. C6 평가 요청 — 감정 없이 순수 eval_request 만 있을 때
    if detect_eval_request(last_user_text):
        if _has_rich_self_disclosure(state):
            return Composition(
                primary_type=MsgType.CONFRONTATION,
                confrontation_block="C6",
                apply_self_pr_prefix=apply_self_pr_prefix,
            )
        # 베타 hotfix (2026-04-28): RANGE_REAFFIRM 막막함 가정 진앙 (6/20).
        # _has_rich_self_disclosure 분류 자체는 유지 (로깅용 / 추후 재활성화 가능성).
        # 자기개시 부족 시 PROBE 로 재라우팅 — 더 정보 끌어내기.
        return Composition(primary_type=MsgType.PROBE)

    # 7. C7 일반화 회피
    if detect_generalization(last_user_text):
        return Composition(
            primary_type=MsgType.CONFRONTATION,
            confrontation_block="C7",
            apply_self_pr_prefix=apply_self_pr_prefix,
        )

    # 8. A-3 other 트리거

    # 8.1. 해명 요청 → obs≥2 이면 DIAGNOSIS, 아니면 OBSERVATION
    if flags.has_explain_req:
        if state.observation_count >= 2:
            return Composition(primary_type=MsgType.DIAGNOSIS)
        return Composition(primary_type=MsgType.OBSERVATION)

    # 8.2. 메타 반박 → META_REBUTTAL (세션 1회)
    if flags.has_meta_challenge and not state.meta_rebuttal_used:
        return Composition(primary_type=MsgType.META_REBUTTAL)

    # 8.3. 근거 불신 → EVIDENCE_DEFENSE (세션 1회)
    if flags.has_evidence_doubt and not state.evidence_defense_used:
        return Composition(primary_type=MsgType.EVIDENCE_DEFENSE)

    # 9. A-9 단답 스트릭 1-2 분기

    if state.trivial_streak == 2:
        # D 경로 — 수용부 + 축 전환
        state.axis_switch_required = True
        return Composition(primary_type=MsgType.OBSERVATION)

    if state.trivial_streak == 1:
        # B 경로 — 같은 축 좁힘
        if last_assistant.msg_type == MsgType.OBSERVATION:
            return Composition(primary_type=MsgType.EXTRACTION)
        return Composition(primary_type=MsgType.PROBE)

    # 9. A-2 비율 강제

    # 9.1. 수집 3연속 → RECOGNITION 강제
    if state.collection_streak >= 3:
        return Composition(
            primary_type=MsgType.RECOGNITION,
            apply_self_pr_prefix=apply_self_pr_prefix,
        )

    # 9.2. RECOGNITION 하한
    if state.observation_count >= 3:
        recog_needed = 2
    elif state.observation_count >= 2:
        recog_needed = 1
    else:
        recog_needed = 0
    recog_done = state.type_counts.get(MsgType.RECOGNITION, 0)
    if (
        (recog_needed - recog_done) > 0
        and state.observations_since_recognition >= 2
    ):
        return Composition(primary_type=MsgType.RECOGNITION)

    # 9.3. DIAGNOSIS 하한 (총 8턴 이상 + obs≥3 + concede)
    total_turns = len(a_turns)
    if total_turns >= 8:
        diag_done = state.type_counts.get(MsgType.DIAGNOSIS, 0)
        diag_needed = max(1, int(total_turns * DIAGNOSIS_MIN_RATIO))
        if (
            diag_done < diag_needed
            and state.observation_count >= 3
            and flags.has_concede
        ):
            return Composition(primary_type=MsgType.DIAGNOSIS)

    # 10. Phase 기본

    # 10.1. concede + obs ≥ 3 → RECOGNITION 먼저, 직전 RECOGNITION 이면 DIAGNOSIS
    if flags.has_concede and state.observation_count >= 3:
        if last_assistant.msg_type == MsgType.RECOGNITION:
            return Composition(primary_type=MsgType.DIAGNOSIS)
        return Composition(primary_type=MsgType.RECOGNITION)

    # 10.2. defensive + obs ≥ 2 → CONFRONTATION (블록 미지정 — Haiku 추론)
    if flags.is_defensive and state.observation_count >= 2:
        return Composition(
            primary_type=MsgType.CONFRONTATION,
            apply_self_pr_prefix=apply_self_pr_prefix,
        )

    # 10.3. obs < 3 → 수집 순환 (같은 타입 2연속 회피)
    if state.observation_count < 3:
        if last_assistant.msg_type == MsgType.OBSERVATION:
            return Composition(primary_type=MsgType.PROBE)
        if last_assistant.msg_type == MsgType.PROBE:
            return Composition(primary_type=MsgType.EXTRACTION)
        return Composition(
            primary_type=MsgType.OBSERVATION,
            apply_self_pr_prefix=apply_self_pr_prefix,
        )

    # 10.4. DIAGNOSIS 직후 → SOFT_WALKBACK
    if last_assistant.msg_type == MsgType.DIAGNOSIS:
        return Composition(primary_type=MsgType.SOFT_WALKBACK)

    # 10.5. fallback
    return Composition(primary_type=MsgType.RECOGNITION)


# ─────────────────────────────────────────────
#  Backward compat
# ─────────────────────────────────────────────

def decide_next_message(state: ConversationState) -> MsgType:
    """primary_type 만 반환 (기존 호출부 호환)."""
    return decide(state).primary_type


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _consecutive_self_disclosure(state: ConversationState, n: int = 2) -> bool:
    """최근 n 개 UserTurn 모두 has_self_disclosure 인지."""
    u_turns = state.user_turns()
    if len(u_turns) < n:
        return False
    return all(t.flags.has_self_disclosure for t in u_turns[-n:])


def _has_rich_self_disclosure(state: ConversationState) -> bool:
    """세션 #7 §2.4 C6 vs RANGE_REAFFIRM 분기 휴리스틱.

    최근 4 UserTurn 중 (a) flags.has_self_disclosure=True OR (b) 50자 이상 발화
    인 턴이 1건 이상이면 "자기개시 풍부" 판정 → C6 경로.
    없으면 "막막함 우세" 로 간주.

    임계치 1로 둔 근거: 장문 자기개시 한 건만 있어도 Sia 가 그 안에서 두 기준
    (서연 fixture "기괴한 순간" vs "원래 선") 을 추출해 C6 재프레임 가능.

    ※ 베타 hotfix (2026-04-28): "막막함 우세" 분류 후 RANGE_REAFFIRM 경로는 폐기.
      decide() / _empathy_combined() 에서 PROBE 로 재라우팅. 분류 자체는 유지.
    """
    u_turns = state.user_turns()
    if not u_turns:
        return False

    for t in u_turns[-4:]:
        if t.flags.has_self_disclosure or len(t.text) >= 50:
            return True
    return False


def _empathy_combined(
    state: ConversationState, flags: UserMessageFlags,
) -> Composition:
    """감정 트리거 시 EMPATHY_MIRROR + secondary 결합 출력 Composition 생성.

    세션 #7 §1.3 (a): 감정 단어 트리거 단독 조건으로 결합 출력 활성화.
    §1.5: 둘째/셋째 문장 = PROBE / OBSERVATION / RECOGNITION.

    Secondary 결정:
      - 직전 유저 발화에 eval_request + 자기개시 풍부 → CONFRONTATION (block=C6)
      - 직전 유저 발화에 eval_request + 막막함 우세 → PROBE (베타 hotfix 후)
      - observation_count >= 3 → RECOGNITION
      - 그 외 → PROBE (default)

    ※ 베타 hotfix (2026-04-28): RANGE_REAFFIRM secondary 폐기 — 막막함 가정 진앙.
      자기개시 부족 시 PROBE (default 와 동일) 로 재라우팅.
    """
    last_user = state.last_user()
    last_text = last_user.text if last_user else ""

    if detect_eval_request(last_text):
        if _has_rich_self_disclosure(state):
            return Composition(
                primary_type=MsgType.EMPATHY_MIRROR,
                secondary_type=MsgType.CONFRONTATION,
                confrontation_block="C6",
            )
        # 베타 hotfix (2026-04-28): RANGE_REAFFIRM 막막함 가정 진앙.
        # 자기개시 부족 + 감정 + 평가 요청 → EMPATHY + PROBE (default 와 일치).
        return Composition(
            primary_type=MsgType.EMPATHY_MIRROR,
            secondary_type=MsgType.PROBE,
        )

    if state.observation_count >= 3:
        return Composition(
            primary_type=MsgType.EMPATHY_MIRROR,
            secondary_type=MsgType.RECOGNITION,
        )

    return Composition(
        primary_type=MsgType.EMPATHY_MIRROR,
        secondary_type=MsgType.PROBE,
    )


# ─────────────────────────────────────────────
#  v4 Turn Flow Decision (T1-T5, 2026-04-28)
#
#  Phase A: T1-T5 의도 가이드 + Haiku 자유 생성. T6+ 미빌드 → None → 페르소나 C fallback.
# ─────────────────────────────────────────────

V4Turn = Literal[
    "T1",
    "T2-A", "T2-C",
    "T3-base", "T3-norm",
    "T4",
    "T5-A", "T5-B",
]

_V4_T2_LENGTH_THRESHOLD = 30


def decide_v4(state: ConversationState, user_flags: UserMessageFlags) -> Optional[V4Turn]:
    """v4 turn flow 라우팅 (T1-T5).

    분기:
      T2: 30자 미만 → T2-A / 30자+ → T2-C
      T3: has_self_doubt → T3-norm / 그 외 → T3-base
      T5: has_uncertainty → T5-B / 그 외 → T5-A

    Returns None for user_turn_count >= 5 → 라우터 페르소나 C fallback.
    """
    user_turn_count = len(state.user_turns())

    if user_turn_count == 0:
        return "T1"
    if user_turn_count == 1:
        last_user = state.last_user()
        last_text = last_user.text if last_user else ""
        return "T2-A" if len(last_text.strip()) < _V4_T2_LENGTH_THRESHOLD else "T2-C"
    if user_turn_count == 2:
        return "T3-norm" if user_flags.has_self_doubt else "T3-base"
    if user_turn_count == 3:
        return "T4"
    if user_turn_count == 4:
        return "T5-B" if user_flags.has_uncertainty else "T5-A"
    return None
