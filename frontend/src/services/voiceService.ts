import type { AiDraftResult, TranscriptionResult } from "@/types";
import { api } from "./http";

/**
 * 음성 진료 초안 서비스 (Facade) — 백엔드 /voice 파이프라인 연동.
 *
 * - transcribe: 녹음 오디오 → STT 전사(whisper 미설치 시 passthrough → 빈 결과).
 * - generateDraft: 전사 → SOAP(근거기반) → Problem List → CDSS Risk → 검증.
 */
class VoiceService {
  /** 오디오 Blob 을 STT 로 전사. multipart/form-data 로 전송한다. */
  async transcribe(audio: Blob): Promise<TranscriptionResult> {
    const form = new FormData();
    form.append("audio", audio, "recording.webm");
    return api.postForm<TranscriptionResult>("/voice/transcribe", form);
  }

  /** 전사 텍스트로 구조화 AI 진료 초안 생성·저장. 검증 실패 시 422 → Error. */
  async generateDraft(patientId: string, transcript: string): Promise<AiDraftResult> {
    return api.post<AiDraftResult>("/voice/draft", { patientId, transcript });
  }

  /** 환자별 초안 이력. */
  async listDrafts(patientId: string): Promise<AiDraftResult[]> {
    return api.get<AiDraftResult[]>(`/voice/drafts/${patientId}`);
  }
}

export const voiceService = new VoiceService();
