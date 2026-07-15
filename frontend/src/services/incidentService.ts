import { api } from "@/services/http";

export interface Incident {
  id: string;
  incidentNo: string;
  status: "OPEN" | "INVESTIGATING" | "RESOLVED" | "SIGNED" | "LOCKED";
  severity: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
  affectedService: string | null;
  moduleName: string;
  endpoint: string | null;
  errorMessage: string;
  stackTrace: string | null;
  occurrenceCount: number;
  firstOccurredAt: string;
  lastOccurredAt: string;
  resolvedAt: string | null;
  rootCause: string | null;
  actionTaken: string | null;
  assignedToUserId: string | null;
  assignedToName: string | null;
  resolvedByUserId: string | null;
  resolvedByName: string | null;
  signaturePath: string | null;
  signedAt: string | null;
  lockedAt: string | null;
  createdAt: string;
  updatedAt: string;
}

const mapIncident = (data: any): Incident => ({
  id: data.id,
  incidentNo: data.incident_no,
  status: data.status,
  severity: data.severity,
  affectedService: data.affected_service,
  moduleName: data.module_name,
  endpoint: data.endpoint,
  errorMessage: data.error_message,
  stackTrace: data.stack_trace,
  occurrenceCount: data.occurrence_count,
  firstOccurredAt: data.first_occurred_at,
  lastOccurredAt: data.last_occurred_at,
  resolvedAt: data.resolved_at,
  rootCause: data.root_cause,
  actionTaken: data.action_taken,
  assignedToUserId: data.assigned_to_user_id,
  assignedToName: data.assigned_to_name,
  resolvedByUserId: data.resolved_by_user_id,
  resolvedByName: data.resolved_by_name,
  signaturePath: data.signature_path,
  signedAt: data.signed_at,
  lockedAt: data.locked_at,
  createdAt: data.created_at,
  updatedAt: data.updated_at,
});

export const incidentService = {
  list: async (status?: string, sortBy: string = "latest"): Promise<Incident[]> => {
    let url = `/incidents?sort_by=${sortBy}`;
    if (status) url += `&status=${status}`;
    const data = await api.get<any[]>(url);
    return data.map(mapIncident);
  },

  get: async (id: string): Promise<Incident> => {
    const data = await api.get<any>(`/incidents/${id}`);
    return mapIncident(data);
  },

  updateStatus: async (id: string, status: string): Promise<Incident> => {
    const data = await api.patch<any>(`/incidents/${id}/status`, { status });
    return mapIncident(data);
  },
  
  assign: async (id: string, assignedToUserId: string, assignedToName: string): Promise<Incident> => {
    const data = await api.patch<any>(`/incidents/${id}/assign`, { assigned_to_user_id: assignedToUserId, assigned_to_name: assignedToName });
    return mapIncident(data);
  },

  resolve: async (id: string, rootCause: string, actionTaken: string): Promise<Incident> => {
    const data = await api.patch<any>(`/incidents/${id}/resolve`, { root_cause: rootCause, action_taken: actionTaken });
    return mapIncident(data);
  },

  analyze: async (id: string): Promise<{ rootCause: string, actionTaken: string }> => {
    const data = await api.post<any>(`/incidents/${id}/analyze`);
    return {
      rootCause: data.root_cause,
      actionTaken: data.action_taken
    };
  },

  sign: async (id: string): Promise<Incident> => {
    const data = await api.post<any>(`/incidents/${id}/sign`, {});
    return mapIncident(data);
  },

  lock: async (id: string): Promise<Incident> => {
    const data = await api.post<any>(`/incidents/${id}/lock`);
    return mapIncident(data);
  },
};
