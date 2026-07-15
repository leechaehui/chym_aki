import type { NotificationInput } from "@/types";
import { EventBus } from "@/lib/eventBus";

/**
 * 도메인 이벤트 정의 (Observer 의 발행 메시지).
 * 협진 요청·병리 결과 등록·AKI 위험도 변경 등 "무슨 일이 일어났는가"만 표현한다.
 * 알림 생성 방법(severity/채널)은 모르며, 아래 eventToNotification 규칙표가 변환을 책임진다.
 */
export type DomainEvent =
  // 신장내과
  | { type: "pathology.resultArrived"; patientName: string; mrn: string }
  | { type: "consult.arrivedNeph"; patientName: string; mrn?: string; consultId?: string }
  | { type: "aki.stage3"; patientName: string; mrn?: string }
  // 병리과
  | { type: "pathology.readSaved"; patientName: string }
  | { type: "consult.requested"; patientName: string; mrn: string; urgency: string; consultId?: string }
  | { type: "consult.urgentRead"; patientName: string; consultId?: string }
  | { type: "incident.akiSpike"; newCases: number };

/** 전역 도메인 이벤트 버스. 발행: 도메인 액션 / 구독: notificationStore.startObserver. */
export const eventBus = new EventBus<DomainEvent>();

/**
 * 병리과 협진 알림의 이동 경로.
 * consultId 가 있으면 해당 협진을 선택하도록 쿼리로 딥링크한다(PathologyWorkspace 가 읽어 선택).
 * 없으면 병리과 화면 자체로만 이동한다. 이미 /pathology 에 있어도 쿼리가 바뀌면 선택이 갱신된다.
 */
function consultLink(consultId?: string): string {
  return consultId ? `/pathology?consult=${consultId}` : "/pathology";
}

/**
 * 신장내과 알림의 이동 경로. mrn 이 있으면 해당 환자를 선택하도록 딥링크한다
 * (NephrologyWorkspace 가 ?patient=<mrn> 을 읽어 선택). 없으면 화면 자체로만 이동.
 */
function nephLink(mrn?: string): string {
  return mrn ? `/nephrology?patient=${mrn}` : "/nephrology";
}

/**
 * 신장 협진(응급/ICU → 신장내과) 알림의 이동 경로. consultId 가 있으면 신장내과 협진
 * 인박스에서 해당 건을 선택하도록 ?consult=<id> 로 딥링크한다(ConsultInbox 가 읽음).
 */
function nephConsultLink(consultId?: string): string {
  return consultId ? `/nephrology?consult=${consultId}` : "/nephrology";
}

/**
 * 응급의학과 알림의 이동 경로. bedId 가 있으면 해당 병상 상세를 열도록 딥링크한다
 * (EmergencyDashboard 가 ?bed=<bedId> 를 읽어 강조·스크롤·상세 모달). 없으면 화면 자체로만 이동.
 */
function bedLink(bedId?: string): string {
  return bedId ? `/emergency?bed=${bedId}` : "/emergency";
}

/**
 * 이벤트 → 알림 변환 규칙표 (Observer 구독자가 사용).
 * 각 이벤트가 어떤 severity·대상 부서·문구의 알림이 되는지 한 곳에서 관리한다.
 * severity 만 정하면 채널(Toast/Banner/Drawer/Modal)은 NotificationFactory 가 결정한다.
 */
export function eventToNotification(e: DomainEvent): NotificationInput {
  switch (e.type) {
    // 신장내과
    case "pathology.resultArrived":
      return { severity: "INFO", department: "nephrology", title: "병리 결과 도착", message: `${e.patientName}(${e.mrn}) 병리 결과가 도착했습니다.`, link: nephLink(e.mrn) };
    case "consult.arrivedNeph":
      return { severity: "ACTION_REQUIRED", department: "nephrology", title: "협진 요청 도착", message: `${e.patientName} 환자에 대한 협진 요청이 도착했습니다. 확인이 필요합니다.`, link: nephConsultLink(e.consultId) };
    case "aki.stage3":
      return { severity: "CRITICAL", department: "nephrology", title: "AKI Stage 3", message: `${e.patientName} 환자 Cr 급상승 — AKI Stage 3, 즉시 평가가 필요합니다.`, link: nephLink(e.mrn) };

    // 병리과
    case "pathology.readSaved":
      return { severity: "INFO", department: "pathology", title: "판독 저장 완료", message: `${e.patientName} 판독 결과가 저장되었습니다.` };
    case "consult.requested":
      return { severity: "ACTION_REQUIRED", department: "pathology", title: "신규 협진 요청", message: `${e.patientName}(${e.mrn}) ${e.urgency} 협진 요청이 접수되었습니다.`, link: consultLink(e.consultId) };
    case "consult.urgentRead":
      return { severity: "CRITICAL", department: "pathology", title: "긴급 판독 요청", message: `${e.patientName} 환자 긴급 판독 요청 — 즉시 확인하세요.`, link: consultLink(e.consultId) };
    case "incident.akiSpike":
      return {
        severity: "WARNING",
        department: "nephrology",
        title: "AKI 발생 스파이크",
        message: `최근 1시간 내 AKI 고위험군이 ${e.newCases}명 급증했습니다. 병동 회진을 서둘러주세요.`,
      };
  }
}
