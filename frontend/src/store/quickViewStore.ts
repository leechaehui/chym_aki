import { create } from "zustand";
import type { PatientIdentity } from "@/types";

interface QuickViewState {
  patient: PatientIdentity | null;
  openQuickView: (patient: PatientIdentity) => void;
  closeQuickView: () => void;
}

export const useQuickViewStore = create<QuickViewState>((set) => ({
  patient: null,
  openQuickView: (patient) => set({ patient }),
  closeQuickView: () => set({ patient: null }),
}));
