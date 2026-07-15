import { create } from "zustand";
import type { Consult, ConsultReply } from "@/types";
import { consultService, type ConsultRequestInput } from "@/services/consultService";

interface ConsultState {
  items: Consult[];
  loading: boolean;
  selectedId: string | null;
  load: () => Promise<void>;
  select: (id: string | null) => void;
  request: (input: ConsultRequestInput) => Promise<Consult>;
  accept: (consultId: string, actor: string) => Promise<void>;
  reply: (consultId: string, reply: ConsultReply) => Promise<void>;
}

/** 협진 전역 상태 — 신장내과/병리과 화면이 공유한다(교차 연동). */
export const useConsultStore = create<ConsultState>((set, get) => ({
  items: [],
  loading: false,
  selectedId: null,

  async load() {
    set({ loading: true });
    const items = await consultService.list();
    set({ items, loading: false });
  },

  select(id) {
    set({ selectedId: id });
  },

  async request(input) {
    const consult = await consultService.request(input);
    set((s) => ({ items: [consult, ...s.items] }));
    return consult;
  },

  async accept(consultId, actor) {
    const target = get().items.find((c) => c.id === consultId);
    if (!target) return;
    const updated = await consultService.accept(target, actor);
    set((s) => ({ items: s.items.map((c) => (c.id === consultId ? updated : c)) }));
  },

  async reply(consultId, reply) {
    const target = get().items.find((c) => c.id === consultId);
    if (!target) return;
    const updated = await consultService.reply(target, reply);
    set((s) => ({ items: s.items.map((c) => (c.id === consultId ? updated : c)) }));
  },
}));
