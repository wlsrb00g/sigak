# SIGAK

**사용자의 현재 이미지와 원하는 이미지를 같은 기준으로 비교하고, 그 차이에 기반한 행동 방향을 제안하는 AI 이미지 컨설팅 프로젝트입니다.**

## Problem

얼굴 이미지에서 얻는 구조적 특징과 인터뷰에서 표현되는 목표 이미지는 형태가 다른 정보입니다. 두 입력을 비교 가능한 표현으로 연결하고, 일관된 추천으로 이어지도록 시스템의 판단 역할을 나눴습니다.

## My Contribution

### 1. Common Representation Design

얼굴 분석으로 얻은 **Current State**와 인터뷰에서 해석한 **Target State**를 같은 표현 체계로 변환해 직접 비교할 수 있도록 구조를 설계했습니다.

### 2. LLM × Deterministic Logic Separation

- **LLM:** 비정형 인터뷰 해석과 결과 설명 생성
- **Python / rules:** current-target gap 및 우선순위 계산, recommendation/action 선택

추천 판단을 전부 LLM에 맡기지 않고, 계산과 설명의 역할을 분리했습니다.

## How It Works

```text
Face image → feature analysis → Current State ─┐
                                                ├→ gap / priority → rule-based action
Interview → LLM interpretation → Target State ┘                         ↓
                                                               LLM explanation / report
```

## Implementation

선별한 코드 스냅샷에는 **InsightFace** 기반 얼굴 분석, 좌표와 gap 계산, 규칙 기반 action specification, LLM 해석·리포트 경로가 포함됩니다. FastAPI backend 모듈과 Next.js UI route 일부를 발췌한 저장소라 전체 앱의 단독 실행 패키지나 완전 통합 배포본은 아닙니다.

## Service

[SIGAK service page](https://www.sigak.asia/) — 서비스 backend는 현재 오프라인입니다. 개인정보가 포함된 실제 얼굴·인터뷰·리포트는 저장소에 포함하지 않았습니다.

## Technical Notes

이 코드는 `Sigak-main.zip`에서 추린 portfolio source snapshot입니다. Antler Korea KOR8 프로젝트의 개인별 수행 범위와 코드 작성자 신원은 코드만으로 확정하지 않습니다. 서비스 개발 기간, 팀 성과 및 재배포 권한은 [`AUDIT_REPORT.md`](AUDIT_REPORT.md)를 참고하세요.
