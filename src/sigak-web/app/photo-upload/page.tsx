/**
 * /photo-upload — Sia 종료 → 사진 업로드 → 옛 SIGAK_V3 PI 시스템 entry.
 *
 * Flow:
 *   /sia → /sia/done (LoadingSlides 15s) → /photo-upload (이 페이지)
 *   → POST /api/v1/submit (사진 + interview JSON)
 *   → POST /api/v1/analyze/{user_id} (AI 파이프라인)
 *   → /report/{report_id}/full redirect
 *
 * BETA 5/15 까지 무료. backend `_is_pi_beta_free()` 가 access_level="full" 강제.
 *
 * 입력:
 *   - query ?from_session={sessionId}  옵션, Sia session reference (현재 미사용)
 *   - localStorage `sigak_user_id`  필수 (Kakao 로그인)
 *
 * 안내 톤 (테스터 피드백 2026-04-27):
 *   - 헤드: "정면 사진 한 장 보여주세요"
 *   - 본문: "얼굴이 또렷하게 잡히는 사진이면 돼요."
 *   - 정확도 가이드 (강조): 정면 / 자연광 / 화장 없음 3 조건. 옵션이 아닌
 *     권장. 얼굴형·피부톤 분석 정확도 직결.
 */

"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import {
  ApiError,
  runAnalysis,
  submitAll,
} from "@/lib/api/client";
import { getCurrentUser } from "@/lib/auth";
import { SigakLoading, TopBar } from "@/components/ui/sigak";
import { TokenInsufficientModal } from "@/components/sigak/token-insufficient-modal";
import { MaleBetaComingSoon } from "@/components/sigak/male-beta-coming-soon";
import { useTokenBalance } from "@/hooks/use-token-balance";
import { useMaleBetaBlock } from "@/hooks/use-male-beta-block";

const MAX_PHOTOS = 3;
// PI 갱신 비용 (2회차+). 1회차는 BETA 무료.
const COST_PI_REGENERATE = 50;

interface PhotoEntry {
  url: string;
  file: File;
}

function PhotoUploadContent() {
  const router = useRouter();
  const params = useSearchParams();
  const fromSession = params.get("from_session");
  // 2026-04-27: ?regenerate=1 = 2회차+ PI 갱신. 50토큰 정책.
  // TODO(backend): /api/v1/submit + /api/v1/analyze 호출 시 regenerate=1 받으면
  //   서버에서도 50토큰 차감 + race condition 검증. 현재는 frontend client-side
  //   잔액 가드만 (race 시 server 402 fallback 패턴 — aspiration 정합).
  const isRegenerate = params.get("regenerate") === "1";

  const { balance } = useTokenBalance();
  // 남성 v1.1 베타 차단 (2026-04-27) — male 풀 미정합 영역
  const { checking: genderChecking, blocked: maleBlocked } = useMaleBetaBlock();

  const [photos, setPhotos] = useState<PhotoEntry[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [stage, setStage] = useState<"idle" | "submit" | "analyze">("idle");
  const [error, setError] = useState<string | null>(null);
  const [showTokenModal, setShowTokenModal] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const replaceIdxRef = useRef<number | null>(null);

  // 인증 가드
  useEffect(() => {
    const user = getCurrentUser();
    if (!user) {
      router.replace("/");
    }
  }, [router]);

  // unmount 시 object URL 해제
  useEffect(() => {
    return () => {
      photos.forEach((p) => URL.revokeObjectURL(p.url));
    };
    // 의도적으로 mount 시점 photos snapshot 만 cleanup. ignore deps lint.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const url = URL.createObjectURL(file);

      if (replaceIdxRef.current !== null) {
        const idx = replaceIdxRef.current;
        setPhotos((prev) => {
          const next = [...prev];
          if (next[idx]) URL.revokeObjectURL(next[idx].url);
          next[idx] = { url, file };
          return next;
        });
        replaceIdxRef.current = null;
      } else {
        setPhotos((prev) => [...prev, { url, file }]);
      }
      e.target.value = "";
    },
    [],
  );

  const handleAdd = useCallback(() => {
    replaceIdxRef.current = null;
    fileRef.current?.click();
  }, []);

  const handleReplace = useCallback((idx: number) => {
    replaceIdxRef.current = idx;
    fileRef.current?.click();
  }, []);

  const handleRemove = useCallback((idx: number) => {
    setPhotos((prev) => {
      const removed = prev[idx];
      if (removed) URL.revokeObjectURL(removed.url);
      return prev.filter((_, i) => i !== idx);
    });
  }, []);

  const handleSubmit = useCallback(async () => {
    if (photos.length === 0) return;
    const user = getCurrentUser();
    if (!user) {
      router.replace("/");
      return;
    }

    // 선결제 가드 (2회차+ 갱신만): client-side 잔액 검증 → 부족 시 모달.
    // server 402 race condition 은 catch 블록에서 동일 모달로 fallback.
    // (aspiration 정합 패턴)
    if (isRegenerate && balance !== null && balance < COST_PI_REGENERATE) {
      setShowTokenModal(true);
      return;
    }

    setSubmitting(true);
    setError(null);
    setStage("submit");

    try {
      const files = photos.map((p) => p.file);

      // Phase 1: /api/v1/submit — 사진 + 최소 interview JSON 저장
      // Sia 대화 결과는 신 vault 에 별도 저장됨. 옛 main.py 는 자체 INTERVIEWS dict 사용.
      // gender 필드 절대 전달 금지 — onboarding/essentials 가 유일한 권위 경로.
      // 옛 SIGAK_V1 시대 hardcode "female" 이 본인 male 을 침식하던 사고 차단 (2026-04-27).
      const submitResult = await submitAll(
        {
          user_id: user.userId,
          name: user.name || "익명",
          tier: "standard",
          ...(fromSession ? { sia_session_id: fromSession } : {}),
        },
        files,
      );

      // Phase 2: /api/v1/analyze/{user_id} — AI 파이프라인 실행 → report_id 회수
      setStage("analyze");
      const analyzeResult = await runAnalysis(submitResult.user_id);

      if (!analyzeResult.report_id) {
        throw new Error("리포트 생성에 실패했어요.");
      }

      // Phase 3: /report/{report_id}/full redirect
      // BETA 우회 분기로 backend 가 access_level="full" 강제 응답 → paywall 미노출.
      router.replace(`/report/${analyzeResult.report_id}/full`);
    } catch (err) {
      setSubmitting(false);
      setStage("idle");
      if (err instanceof ApiError) {
        if (err.status === 401) {
          router.replace("/");
          return;
        }
        if (err.status === 402) {
          // server 잔액 부족 (race condition) → 모달 노출 (client check 통과 후 충돌 케이스)
          setShowTokenModal(true);
          return;
        }
        if (err.status === 409) {
          // 남성 v1.1 차단 안전망 (2026-04-27)
          // 보통 useMaleBetaBlock 으로 진입 차단되지만 race / 가드 누락 시 fallback.
          setError(
            err.message ||
            "남성 회원님들을 위한 시각 비밀 레포트는 준비중입니다.",
          );
          return;
        }
        setError(err.message);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("제출 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요.");
      }
    }
  }, [photos, router, fromSession, isRegenerate, balance]);

  // 남성 v1.1 베타 차단 — gender 확인 중에는 로딩, male 이면 안내 화면
  if (genderChecking) {
    return <SigakLoading message="" />;
  }
  if (maleBlocked) {
    return <MaleBetaComingSoon featureName="시각 비밀 레포트" />;
  }

  // 진행 중 — SigakLoading 단일 표준 (2026-04-27 마케터 1815 정합)
  if (submitting) {
    return <SigakLoading message="sia가 피드를 분석중이에요." />;
  }

  return (
    <>
      <TopBar backTarget="/" />
      <main
        className="animate-fade-in"
        style={{
          minHeight: "calc(100vh - 64px)",
          background: "var(--color-paper)",
          color: "var(--color-ink)",
          display: "flex",
          flexDirection: "column",
          padding: "32px 28px 40px",
        }}
      >
      <section
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          maxWidth: 380,
          width: "100%",
          marginInline: "auto",
        }}
      >
        <span
          className="font-sans uppercase"
          style={{
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: "1.5px",
            opacity: 0.4,
          }}
        >
          {isRegenerate ? "PHOTO · 갱신 · 50토큰" : "PHOTO"}
        </span>
        <h1
          className="font-serif"
          style={{
            marginTop: 16,
            fontSize: 28,
            fontWeight: 400,
            lineHeight: 1.4,
            letterSpacing: "-0.01em",
          }}
        >
          정면 사진 한 장<br />
          보여주세요.
        </h1>
        <p
          className="font-sans"
          style={{
            marginTop: 16,
            fontSize: 13,
            lineHeight: 1.7,
            letterSpacing: "-0.005em",
            opacity: 0.6,
          }}
        >
          얼굴이 또렷하게 잡히는 사진이면 돼요.
        </p>

        {/* 정확도 가이드 — 얼굴형·피부톤 분석 직결 (테스터 피드백 2026-04-27)
            "정면 / 자연광 / 화장 없음" 3 조건 시각화. 옵션이 아닌 권장 (강한 톤). */}
        <div
          className="font-sans"
          style={{
            marginTop: 20,
            padding: "16px 18px",
            border: "1px solid var(--color-line)",
            borderRadius: 10,
            background: "rgba(0, 0, 0, 0.025)",
          }}
        >
          <p
            className="uppercase"
            style={{
              margin: 0,
              marginBottom: 10,
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              letterSpacing: "0.14em",
              opacity: 0.55,
            }}
          >
            정확한 분석을 위해
          </p>
          <ul
            style={{
              margin: 0,
              padding: 0,
              listStyle: "none",
              display: "flex",
              flexDirection: "column",
              gap: 7,
              fontSize: 13,
              lineHeight: 1.55,
              letterSpacing: "-0.005em",
            }}
          >
            <li style={{ display: "flex", gap: 8 }}>
              <span aria-hidden style={{ opacity: 0.45 }}>·</span>
              <span>
                <strong style={{ fontWeight: 600 }}>정면 사진</strong> — 얼굴이 카메라를 똑바로 보는 각도
              </span>
            </li>
            <li style={{ display: "flex", gap: 8 }}>
              <span aria-hidden style={{ opacity: 0.45 }}>·</span>
              <span>
                <strong style={{ fontWeight: 600 }}>자연광</strong> — 창가·야외 권장 (실내 조명·플래시는 피부톤 왜곡)
              </span>
            </li>
            <li style={{ display: "flex", gap: 8 }}>
              <span aria-hidden style={{ opacity: 0.45 }}>·</span>
              <span>
                <strong style={{ fontWeight: 600 }}>화장 없는 상태</strong> — 피부톤·얼굴형 정확도 직결
              </span>
            </li>
          </ul>
        </div>

        {/* 사진 슬롯 그리드 — 정면 1 + 측면 2 (마케터 dashed + 슬롯 번호 차용) */}
        <div
          style={{
            marginTop: 32,
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: 10,
          }}
        >
          {Array.from({ length: MAX_PHOTOS }, (_, i) => {
            const photo = photos[i];
            const slotLabels = ["정면", "측면 1", "측면 2"];
            const slotNum = `0${i + 1}`;
            const filled = !!photo;
            return (
              <div
                key={i}
                style={{
                  aspectRatio: "3 / 4",
                  borderRadius: 12,
                  border: filled
                    ? "1.5px solid var(--color-line-strong)"
                    : "1.5px dashed var(--color-line-strong)",
                  background: filled ? "transparent" : "rgba(0, 0, 0, 0.04)",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  overflow: "hidden",
                  position: "relative",
                  transition: "border-color 0.2s ease, background 0.2s ease",
                }}
              >
                {/* 슬롯 번호 (좌상단) */}
                <span
                  style={{
                    position: "absolute",
                    top: 10,
                    left: 12,
                    fontFamily: "var(--font-mono)",
                    fontSize: 10,
                    letterSpacing: "0.14em",
                    color: filled ? "rgba(255,255,255,0.85)" : "var(--color-mute-2)",
                    textShadow: filled ? "0 1px 2px rgba(0,0,0,0.4)" : "none",
                    zIndex: 1,
                  }}
                >
                  {slotNum}
                </span>

                {photo ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={photo.url}
                      alt={slotLabels[i]}
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                        borderRadius: 10,
                      }}
                    />
                    <button
                      type="button"
                      onClick={() => handleRemove(i)}
                      className="font-sans"
                      aria-label={`${slotLabels[i]} 사진 제거`}
                      style={{
                        position: "absolute",
                        top: 7,
                        right: 7,
                        width: 22,
                        height: 22,
                        borderRadius: "50%",
                        border: 0,
                        background: "rgba(45,45,45,0.65)",
                        color: "#fff",
                        fontSize: 12,
                        cursor: "pointer",
                      }}
                    >
                      ✕
                    </button>
                    <button
                      type="button"
                      onClick={() => handleReplace(i)}
                      className="font-sans"
                      style={{
                        position: "absolute",
                        bottom: 6,
                        left: 6,
                        right: 6,
                        padding: "4px 0",
                        borderRadius: 6,
                        border: 0,
                        background: "rgba(45,45,45,0.65)",
                        color: "#fff",
                        fontSize: 10,
                        letterSpacing: "0.3px",
                        cursor: "pointer",
                      }}
                    >
                      재촬영
                    </button>
                  </>
                ) : (
                  <button
                    type="button"
                    onClick={handleAdd}
                    className="font-sans"
                    aria-label={`${slotLabels[i]} 추가`}
                    style={{
                      width: "100%",
                      height: "100%",
                      border: 0,
                      background: "transparent",
                      color: "var(--color-ink)",
                      cursor: "pointer",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 8,
                    }}
                  >
                    <span style={{ fontSize: 22, color: "var(--color-mute-2)", lineHeight: 1 }}>
                      +
                    </span>
                    <span
                      style={{
                        fontSize: 11,
                        color: "var(--color-mute)",
                        letterSpacing: "-0.005em",
                      }}
                    >
                      {slotLabels[i]}
                    </span>
                  </button>
                )}
              </div>
            );
          })}
        </div>

        {/* count-dot 3개 (마케터 차용) */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            gap: 8,
            marginTop: 14,
          }}
          aria-hidden
        >
          {Array.from({ length: MAX_PHOTOS }, (_, i) => (
            <span
              key={i}
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background:
                  i < photos.length ? "var(--color-ember)" : "var(--color-line-strong)",
                transition: "background 0.2s ease",
              }}
            />
          ))}
        </div>

        {/* Phase B-7.1 (PI-REVIVE 2026-04-26): capture 속성 제거.
            본인 모바일 피드백: "PI 레포트는 사진 찍게하는데 저건 말도안되고".
            capture="environment" 가 카메라 강제 → 제거 시 OS 피커가
            갤러리/카메라/파일 모두 선택지로 노출. */}
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />

        {error && (
          <p
            role="alert"
            className="font-sans"
            style={{
              marginTop: 16,
              fontSize: 12,
              color: "var(--color-danger)",
              letterSpacing: "-0.005em",
            }}
          >
            {error}
          </p>
        )}
      </section>

      {/* 제출 영역 — 마케터 pill CTA + BETA 무료 라벨 */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 4,
          marginInline: "auto",
          maxWidth: 380,
          width: "100%",
          marginTop: 28,
        }}
      >
        <button
          type="button"
          onClick={handleSubmit}
          disabled={photos.length === 0 || submitting}
          className="font-sans"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            width: "100%",
            padding: "17px 24px",
            border: 0,
            borderRadius: 100,
            background:
              photos.length === 0 ? "var(--color-line-strong)" : "var(--color-ink)",
            color: "var(--color-paper)",
            fontSize: 15,
            fontWeight: 600,
            letterSpacing: "-0.012em",
            cursor:
              photos.length === 0 || submitting ? "not-allowed" : "pointer",
            transition: "all 0.25s ease",
          }}
        >
          {photos.length === 0
            ? "사진을 1장 이상 올려주세요"
            : "정밀 분석 시작하기 →"}
        </button>
      </div>
    </main>
    {/* PI 갱신 잔액 부족 모달 (2회차+ 50토큰 정책) */}
    <TokenInsufficientModal
      open={showTokenModal}
      balance={balance ?? 0}
      required={COST_PI_REGENERATE}
      onCharge={() => router.push("/tokens/purchase?intent=pi_regenerate")}
      onClose={() => setShowTokenModal(false)}
      title="시각 비밀 레포트 갱신"
      sub={`갱신에는 ${COST_PI_REGENERATE} 토큰이 필요해요.`}
    />
    </>
  );
}

export default function PhotoUploadPage() {
  return (
    <Suspense fallback={<SigakLoading message="잠시만요" hint="" />}>
      <PhotoUploadContent />
    </Suspense>
  );
}
