import type { Bed, BedPatientDetail, BedZone } from "@/types";
import { api } from "./http";

/** 구역별 병상 집계 결과. */
export interface BedSummary {
  zone: BedZone;
  total: number;
  occupied: number;
  available: number;
}

/** 병상 배정/해제 결과(백엔드). */
interface BedActionResult {
  bed: Bed;
  admission: unknown | null;
}

/**
 * 병상/응급환자 서비스 (Facade) — FastAPI 연동.
 *
 * listBeds / listEmergencyPatients / listBedDetails / listReservations 는 백엔드에서 조회한다.
 * 상세·예약은 bedId → 값 맵으로 한 번에 받아, 보드 렌더 중 동기 조회(detailOf/reservationOf)에
 * 사용할 수 있도록 컴포넌트가 캐시한다.
 * summarize 는 이미 조회한 병상 배열에 대한 순수 집계(추가 요청 없음).
 */
class BedService {
  async listBeds(): Promise<Bed[]> {
    return api.get<Bed[]>("/beds");
  }

/** 사용중 병상별 입실 상세(처방·검사·AKI 처치권고). bedId → 상세. */
  async listBedDetails(): Promise<Record<string, BedPatientDetail>> {
    return api.get<Record<string, BedPatientDetail>>("/beds/details");
  }

  /** 병상 배정(트랜잭션) — 응급의학과/관리자. */
  async assign(
    bedId: string,
    payload: { patientId?: string; patientName?: string; sex?: string; age?: number; diagnosis?: string },
  ): Promise<Bed> {
    const res = await api.post<BedActionResult>(`/beds/${bedId}/assign`, payload);
    return res.bed;
  }

  /** 병상 해제(트랜잭션) — 응급의학과/관리자. */
  async release(bedId: string): Promise<Bed> {
    const res = await api.post<BedActionResult>(`/beds/${bedId}/release`, {});
    return res.bed;
  }

  /** 구역별 총/사용중/잔여 집계 — KPI·보드 헤더에 사용(순수 계산). */
  summarize(beds: Bed[], zone: BedZone): BedSummary {
    const z = beds.filter((b) => b.zone === zone);
    const occupied = z.filter((b) => b.state === "occupied").length;
    const available = z.filter((b) => b.state === "available").length;
    return { zone, total: z.length, occupied, available };
  }
}

export const bedService = new BedService();
