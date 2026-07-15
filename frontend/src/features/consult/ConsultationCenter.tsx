import { PageHeader } from "@/components/common/PageHeader";
import { ConsultInbox } from "./ConsultInbox";

/**
 * 협진 센터 — 신장내과 협진 전용 독립 페이지(메인 진료 화면에서 분리).
 * 응급의학과/ICU 수신 협진(처리)과 병리과로 보낸 협진(추적)을 한 곳에서 본다.
 */
export function ConsultationCenter() {
  return (
    <div className="w-full">
      <PageHeader title="협진 센터" subtitle="응급 협진 수신·회신 · 병리 협진 현황 추적" />

      {/* 응급의학과/ICU → 신장내과 응급 협진 인박스(접수·회신) */}
      <div className="mb-4">
        <ConsultInbox
          kind="nephrology"
          mode="receiver"
          title="응급 협진 요청 (ICU/ER)"
          emptyTitle="수신된 응급 협진이 없습니다"
          emptyDescription="응급의학과에서 AKI 응급 협진을 보내면 여기에 표시됩니다."
        />
      </div>

      {/* 병리과로 보낸 협진의 상태·회신 추적(요청자 측·읽기 전용) */}
      <ConsultInbox
        kind="pathology"
        mode="requester"
        title="병리 협진 현황 (→ 병리과)"
        emptyTitle="보낸 병리 협진이 없습니다"
        emptyDescription="환자 요약의 '병리 협진 요청'으로 의뢰하면 여기에서 상태와 회신을 추적합니다."
      />
    </div>
  );
}
