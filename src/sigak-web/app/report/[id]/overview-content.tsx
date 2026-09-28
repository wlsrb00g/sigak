"use client";

// 오버뷰 랜딩 콘텐츠 (클라이언트 컴포넌트)
// - free + standard 섹션: 전체 렌더링 (₩5,000 결제 콘텐츠)
// - full 섹션: teaser만 표시
// - CTA: 풀 업그레이드 버튼
// - 공유 버튼 (카카오톡 + 링크 복사)

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import type { ReportData, ReportSection } from "@/lib/types/report";
import { requestUpgrade, getKakaoLoginUrl } from "@/lib/api/client";
import { SectionRenderer } from "@/components/report/section-renderer";
import { ShareButtons } from "@/components/report/share-buttons";
import { TossPaymentFlow } from "@/components/report/toss-payment-flow";

interface OverviewContentProps {
  report: ReportData;
  reportId: string;
}

// 섹션 ID → 한글 라벨 매핑
const SECTION_LABELS: Record<string, string> = {
  hair_recommendation: "헤어 추천",
  action_plan: "액션 플랜",
  type_reference: "유형 레퍼런스",
  celeb_reference: "셀럽 레퍼런스",
  trend_context: "트렌드 컨텍스트",
};

// teaser에서 헤드라인 텍스트 추출
function getTeaserText(section: ReportSection): string | null {
  if (!section.teaser) return null;
  if ("headline" in section.teaser && section.teaser.headline) {
    return section.teaser.headline as string;
  }
  if ("categories" in section.teaser && Array.isArray(section.teaser.categories)) {
    return (section.teaser.categories as string[]).join(" · ");
  }
  return null;
}

export function OverviewContent({ report, reportId }: OverviewContentProps) {
  const router = useRouter();
  const [upgradeLoading, setUpgradeLoading] = useState(false);
  const [tossOrderId, setTossOrderId] = useState<string | null>(null);

  // 소유권 검증: 이미 로그인된 본인 리포트일 때만 컨텍스트 보존
  // 미로그인 or 타인 리포트 → localStorage 건드리지 않음 (공유 링크 보안)
  useEffect(() => {
    const currentUserId = localStorage.getItem("sigak_user_id");
    if (currentUserId && currentUserId === report.user_id) {
      if (report.user_name && report.user_name !== "회원") {
        localStorage.setItem("sigak_user_name", report.user_name);
      }
    }
  }, [report.user_id, report.user_name]);

  // 소유권 확인: 비로그인 → 로그인, 타인 → 차단
  const isOwner = useCallback(() => {
    const currentUserId = localStorage.getItem("sigak_user_id");
    if (!currentUserId) return "not_logged_in" as const;
    if (report.user_id && currentUserId !== report.user_id) return "not_owner" as const;
    return "ok" as const;
  }, [report.user_id]);

  const redirectToLogin = useCallback(async () => {
    sessionStorage.setItem("sigak_redirect", `/report/${reportId}`);
    const { auth_url } = await getKakaoLoginUrl();
    window.location.href = auth_url;
  }, [reportId]);

  // 풀 업그레이드 — 주문 생성 + 결제 플로우 분기
  const handleUpgrade = useCallback(async () => {
    const ownership = isOwner();
    if (ownership === "not_logged_in") { redirectToLogin(); return; }
    if (ownership === "not_owner") {
      alert("본인의 리포트만 업그레이드할 수 있습니다.");
      return;
    }
    setUpgradeLoading(true);
    try {
      const res = await requestUpgrade(reportId);
      if (res.payment_info) {
        const p = res.payment_info;
        const paywall = report.paywall?.full;

        // 토스 위젯 결제: 주문 ID 저장 → 위젯 표시
        if (paywall?.method === "auto" && res.order_id) {
          setTossOrderId(res.order_id);
          setUpgradeLoading(false);
          return;
        }

        // 수동 결제: 결제 안내 페이지 이동
        const params = new URLSearchParams({
          order_id: res.order_id || "",
          amount: String(p.amount),
          bank: p.bank,
          account: p.account,
          holder: p.holder,
          toss: (p as unknown as Record<string, string>).toss_deeplink || "",
          kakao: (p as unknown as Record<string, string>).kakao_deeplink || "",
        });
        router.push(`/questionnaire/payment?${params.toString()}`);
        return;
      }
      if (res.status === "already_full") {
        window.location.reload();
      }
    } catch (e) {
      console.error("[upgrade]", e);
      alert("주문 생성에 실패했습니다. 잠시 후 다시 시도해주세요.");
      setUpgradeLoading(false);
    }
  }, [reportId, report.paywall, router, isOwner, redirectToLogin]);

  // 이미 풀 결제 완료된 경우
  const isFullPaid = ["full_pending", "full"].includes(report.access_level);

  // free + standard 섹션 (전체 렌더링 대상)
  const visibleSections = report.sections.filter(
    (s) => !s.unlock_level || s.unlock_level === "standard"
  );

  // full 섹션 (teaser만)
  const fullSections = report.sections.filter((s) => s.unlock_level === "full");

  const userName = report.user_name || "회원";
  const summarySection = report.sections.find((s) => s.id === "executive_summary");
  const summaryText =
    (summarySection?.content as { summary?: string })?.summary || "";

  return (
    <div className="max-w-2xl mx-auto px-[var(--spacing-page-x-mobile)] md:px-[var(--spacing-page-x)]">
      {/* 1. free + standard 섹션 전체 렌더링 — Phase B-2.5: overlay prop 제거 */}
      {visibleSections.map((section) => (
        <SectionRenderer
          key={section.id}
          section={section}
          accessLevel="standard"
        />
      ))}

      {/* 2. 공유 버튼 */}
      <div className="py-8 border-b border-[var(--color-line)]">
        <ShareButtons
          title={`${userName}님의 시각 리포트`}
          description={summaryText.length > 80 ? summaryText.slice(0, 80) + "..." : summaryText}
        />
      </div>

      {/* 3. full 섹션 teaser 목록 */}
      {fullSections.length > 0 && !isFullPaid && (
        <section className="py-10">
          <h2 className="text-xs font-semibold tracking-[4px] uppercase text-[var(--color-muted)] mb-8">
            FULL REPORT
          </h2>
          <div className="flex flex-col gap-4">
            {fullSections.map((section) => {
              const teaser = getTeaserText(section);
              return (
                <div
                  key={section.id}
                  className="flex items-start gap-4 py-4 border-b border-[var(--color-line)]"
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-muted)] mt-2 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold mb-0.5 text-[var(--color-muted)]">
                      {SECTION_LABELS[section.id] || section.id}
                    </p>
                    {teaser && (
                      <p className="text-xs text-[var(--color-muted)] opacity-60 truncate">
                        {teaser}
                      </p>
                    )}
                  </div>
                  <svg
                    className="w-4 h-4 text-[var(--color-muted)] opacity-40 mt-1 shrink-0"
                    viewBox="0 0 16 16"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                  >
                    <rect x="3" y="7" width="10" height="7" rx="1" />
                    <path d="M5 7V5a3 3 0 016 0v2" />
                  </svg>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* 4. CTA */}
      <section className="py-10 border-t border-[var(--color-line)]">
        {isFullPaid ? (
          <div className="flex flex-col items-center gap-4">
            <p className="text-sm text-[var(--color-muted)]">
              풀 리포트가 열려 있습니다
            </p>
            <button
              onClick={() => {
                const ownership = isOwner();
                if (ownership === "not_logged_in") { redirectToLogin(); return; }
                if (ownership === "not_owner") {
                  alert("본인의 리포트만 열람할 수 있습니다.");
                  return;
                }
                router.push(`/report/${reportId}/full`);
              }}
              className="inline-flex items-center justify-center px-8 py-3.5 text-lg font-medium bg-[var(--color-fg)] text-[var(--color-bg)] hover:opacity-90 transition-colors"
            >
              풀 리포트 보기
            </button>
          </div>
        ) : tossOrderId ? (
          /* 토스 위젯 결제 모드 — 주문 생성 완료 후 위젯 표시 */
          <div className="flex flex-col items-center gap-4">
            <p className="text-2xl font-serif font-bold text-center leading-snug">
              결제 수단을 선택해주세요
            </p>
            <TossPaymentFlow
              orderId={tossOrderId}
              orderName="시각 풀 리포트"
              amount={report.paywall?.full?.price || 0}
            />
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <p className="text-2xl font-serif font-bold text-center leading-snug">
              맞춤 헤어 · 액션 플랜 · 캐스팅까지
            </p>
            <p className="text-sm text-[var(--color-muted)] text-center max-w-xs">
              헤어 추천, 메이크업 액션 플랜, 유형 레퍼런스에
              캐스팅 풀 등록 기회까지 — 나만의 스타일 가이드를 완성하세요
            </p>
            {report.paywall?.full?.original_price && (
              <p className="text-sm text-[var(--color-muted)] line-through">
                ₩{report.paywall.full.original_price.toLocaleString()}
              </p>
            )}
            <button
              onClick={handleUpgrade}
              disabled={upgradeLoading}
              className="inline-flex items-center justify-center px-8 py-3.5 text-lg font-medium bg-[var(--color-fg)] text-[var(--color-bg)] hover:opacity-90 transition-colors mt-2 disabled:opacity-50"
            >
              {upgradeLoading
                ? "주문서 생성 중..."
                : report.paywall?.full
                  ? `₩${report.paywall.full.price.toLocaleString()} 풀 리포트 열기`
                  : "풀 리포트 보기"}
            </button>
            {report.paywall?.full?.total_note && (
              <p className="text-[10px] text-[var(--color-muted)]">
                {report.paywall.full.total_note}
              </p>
            )}
          </div>
        )}
      </section>

      <div className="h-10" />
    </div>
  );
}
