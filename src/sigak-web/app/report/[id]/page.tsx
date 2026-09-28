// 리포트 오버뷰 페이지 (공유용 랜딩)
// - OG 메타태그로 카카오톡/SNS 미리보기 지원
// - cover + executive_summary 전체 공개
// - 나머지 섹션은 teaser(1줄 헤드라인)만 표시
// - CTA: 잠금 해제 버튼 → /report/[id]/full 이동
// - 공유 버튼 (카카오톡 + 링크 복사)

import type { Metadata } from "next";
import Link from "next/link";
import { getReportServerSide } from "@/lib/api/client";
import { OverviewContent } from "./overview-content";
import { ReportNav } from "@/components/report/report-nav";

interface ReportPageProps {
  params: Promise<{ id: string }>;
}

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://sigak.asia";

// 동적 OG 메타태그 — 카카오톡/인스타/트위터 공유 미리보기
export async function generateMetadata({ params }: ReportPageProps): Promise<Metadata> {
  const { id } = await params;
  const report = await getReportServerSide(id);

  if (!report) {
    return {
      title: "SIGAK - 리포트",
      description: "이목구비 비율 · 얼굴형 정밀 분석",
    };
  }

  const userName = report.user_name || "회원";
  const summarySection = report.sections.find((s) => s.id === "executive_summary");
  const summary = (summarySection?.content as { summary?: string })?.summary || "";
  const description = summary.length > 100 ? summary.slice(0, 100) + "..." : summary;

  return {
    title: `${userName}님의 시각 리포트`,
    description: description || "AI 이목구비 분석 · 퍼스널 스타일링 리포트",
    openGraph: {
      title: `${userName}님의 시각 리포트`,
      description: description || "AI 이목구비 분석 · 퍼스널 스타일링 리포트",
      url: `${SITE_URL}/report/${id}`,
      siteName: "SIGAK",
      type: "article",
    },
    twitter: {
      card: "summary",
      title: `${userName}님의 시각 리포트`,
      description: description || "AI 이목구비 분석 · 퍼스널 스타일링 리포트",
    },
  };
}

export default async function ReportPage({ params }: ReportPageProps) {
  const { id } = await params;
  const report = await getReportServerSide(id);

  return (
    <main className="min-h-screen bg-[var(--color-bg)]">
      {/* 네비게이션 */}
      <ReportNav />

      {report ? (
        <div className="pt-4 pb-20">
          <OverviewContent report={report} reportId={id} />
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center min-h-[60vh] px-6 text-center">
          <h1 className="font-[family-name:var(--font-serif)] text-[24px] font-normal mb-3">
            리포트를 불러올 수 없습니다
          </h1>
          <p className="text-[13px] opacity-50 mb-8 max-w-[360px]">
            분석이 아직 완료되지 않았거나, 유효하지 않은 링크입니다.
            잠시 후 다시 시도해 주세요.
          </p>
          <Link
            href="/sia"
            className="inline-block px-8 py-3 text-sm font-bold bg-[var(--color-fg)] text-[var(--color-bg)] transition-opacity duration-200 hover:opacity-85"
          >
            처음으로 돌아가기
          </Link>
        </div>
      )}
    </main>
  );
}
