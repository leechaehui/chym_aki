import { api } from "./http";

export interface DemoResponse {
  status: string;
  message: string;
}

export const demoService = {
  setup: () => api.post<DemoResponse>("/demo/setup"),
  advanceHour: () => api.post<DemoResponse>("/demo/advance-hour"),
  triggerEvent: () => api.post<DemoResponse>("/demo/trigger-event"),
  reset: () => api.post<DemoResponse>("/demo/reset"),
};
