# SIGAK

얼굴 이미지에서 얻은 구조적 특징과 인터뷰에서 얻은 목표 이미지를 하나의 표현 체계로 연결하고, 현재-목표 차이를 바탕으로 추천을 구성하는 AI 개인 이미지 컨설팅 프로젝트입니다.

## 시스템 설계 (선별한 코드 스냅샷 기준)

```text
face image → landmark / structural features → current-state coordinates ┐
interview → LLM interpretation → target-state coordinates              ├→ deterministic gap / priority / action selection
                                                                       └→ LLM-generated explanation and report
```

코드에는 InsightFace 기반 얼굴 분석, 좌표/갭 계산, 규칙 기반 action specification 및 별도 LLM 해석·리포트 생성 경로가 있습니다. 따라서 “LLM이 모든 추천을 자유 생성”하는 것으로 설명하지 않습니다. 서로 다른 파일/서비스 버전이 공존하므로 이 그림은 모든 버전이 단일 배포 파이프라인으로 통합·운영됐다는 증거가 아닙니다.

## 포함 파일과 실행

`src/`에는 `Sigak-main.zip`에서 고른 backend pipeline/service 모듈과 upload/report UI route 일부만 발췌했습니다. 각 route는 대량의 미포함 frontend/backend 모듈에 의존하므로 이 저장소는 바로 실행 가능한 앱이 아닙니다. **실행 명령은 제공하지 않습니다.** API key나 유료 API 호출 없이 정적 소스만 검토했고, 서버를 재가동하거나 결제/API를 호출하지 않았습니다.

## 한계와 상태

사용자 제공 현황상 서비스 서버는 비용 문제로 중단되어 있습니다. 실제 고객 report, 얼굴, 사용자 입력, demo data, 화면 캡처는 개인정보와 권한 문제로 포함하지 않았습니다. Antler KOR8 관련 수행 기간/성과 및 개인별 기여는 공개 가능한 별도 증빙을 확인하기 전까지 단정하지 않습니다. 코드만으로 저자/기여자를 확정할 수 없습니다.

이 선별 스냅샷은 `Desktop/Sigak-main.zip`에서 나온 것으로, 기존 `창업관련/SIGAK_DESIGNBOOK.jsx` 및 이전 `03_SIGAK_PORTFOLIO.zip`과 버전 일치 여부를 완전히 확인하지 못했습니다. repository 공개 전에 팀/회사 IP, 제3자 라이브러리 및 디자인의 재배포 권한을 확인해야 합니다. 라이선스는 추가하지 않았습니다.
