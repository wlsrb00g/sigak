# 공개 저장소 선별 감사 — SIGAK

- 기준 소스: `Desktop/Sigak-main.zip` (약 218 MB); 전체 압축 해제·복사 대신 관련 모듈과 두 UI route만 whitelist 추출함.
- 포함: face/coordinate/LLM/action-spec pipeline, PI face/methodology와 SIA decision/LLM 서비스, upload/report route 일부, backend requirements.
- 제외: archive 전체 `.claude`/`.codex`/history, `.env`/credentials, 사용자 report/data, 얼굴/작품 이미지, mock customer/demo JSON, DB/cache/build output, 전체 Next app 및 대형 `main.py`.
- 코드 확인: backend `pipeline/face.py`, `coordinate.py`, `llm.py`, `action_spec.py`; service `pi_face_pipeline.py`, `pi_methodology.py`, `sia_decision.py`, `sia_llm.py`. UI route는 `@/lib`, components 등 제외된 모듈을 참조하므로 standalone build는 불가.
- 상태: 사용자 진술상 서버 중단. 실행·외부 API·결제·배포는 수행하지 않음.
- 버전/기여: archive 내 여러 generation 존재. 개인의 독점적 기여를 git metadata나 코드만으로 확정하지 않음. public release 권한은 팀/회사와 확인 필요.
