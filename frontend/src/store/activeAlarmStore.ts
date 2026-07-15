import { create } from "zustand";

interface ActiveAlarmState {
  /** 현재 활성(미처리) 알람이 있는 stayId 집합 */
  activeStayIds: Set<number>;
  dismissAlarm: (stayId: number) => void;
  addAlarm: (stayId: number) => void;
}

export const useActiveAlarmStore = create<ActiveAlarmState>((set) => ({
  activeStayIds: new Set(),
  addAlarm: (stayId) =>
    set((s) => ({ activeStayIds: new Set(s.activeStayIds).add(stayId) })),
  dismissAlarm: (stayId) =>
    set((s) => {
      const next = new Set(s.activeStayIds);
      next.delete(stayId);
      return { activeStayIds: next };
    }),
}));
