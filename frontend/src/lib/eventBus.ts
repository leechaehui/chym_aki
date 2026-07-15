/**
 * 경량 EventBus (Observer 패턴).
 * 발행자(도메인 액션)는 publish 로 이벤트를 흘리고, 구독자(알림 생성기 등)는 subscribe 로 관찰한다.
 * 발행자와 구독자가 서로를 모르므로 결합도가 낮다(개방-폐쇄: 구독자 추가가 발행부에 영향 없음).
 */
export type Listener<E> = (event: E) => void;

export class EventBus<E> {
  private listeners = new Set<Listener<E>>();

  /** 구독 등록. 반환한 함수를 호출하면 구독 해제. */
  subscribe(listener: Listener<E>): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /** 이벤트 발행 — 모든 구독자에게 전달. */
  publish(event: E): void {
    this.listeners.forEach((l) => l(event));
  }
}
