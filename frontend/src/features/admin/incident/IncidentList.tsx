import { useEffect, useState } from "react";
import { AlertCircle, FileText, CheckCircle2, Lock, Clock, Search } from "lucide-react";
import { type Incident, incidentService } from "@/services/incidentService";
import { formatDateTime } from "@/lib/format";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { IncidentDetail } from "./IncidentDetail";

const STATUS_CONFIG = {
  OPEN: { label: "신규", icon: AlertCircle, color: "text-red-500 bg-red-50 border-red-200" },
  INVESTIGATING: { label: "원인 파악 중", icon: Search, color: "text-amber-500 bg-amber-50 border-amber-200" },
  RESOLVED: { label: "조치 중 (서명 대기)", icon: CheckCircle2, color: "text-amber-600 bg-amber-50 border-amber-200" },
  SIGNED: { label: "서명 완료", icon: FileText, color: "text-purple-500 bg-purple-50 border-purple-200" },
  LOCKED: { label: "승인 완료", icon: Lock, color: "text-slate-500 bg-slate-50 border-slate-200" },
};

const SEVERITY_COLOR = {
  CRITICAL: "text-red-600 bg-red-100",
  HIGH: "text-orange-600 bg-orange-100",
  MEDIUM: "text-amber-600 bg-amber-100",
  LOW: "text-blue-600 bg-blue-100",
};

export function IncidentList() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null);

  const loadIncidents = async () => {
    setLoading(true);
    try {
      const data = await incidentService.list();
      setIncidents(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadIncidents();
  }, []);

  if (selectedIncidentId) {
    return (
      <IncidentDetail 
        incidentId={selectedIncidentId} 
        onBack={() => {
          setSelectedIncidentId(null);
          loadIncidents();
        }} 
      />
    );
  }

  if (loading) return <div className="p-8 text-center text-slate-500">데이터 불러오는 중...</div>;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>운영 장애 현황</CardTitle>
        <Button variant="outline" size="sm" onClick={loadIncidents}>새로고침</Button>
      </CardHeader>
      <CardContent>
        {incidents.length === 0 ? (
          <div className="text-center py-12 text-slate-500 border border-dashed rounded-lg">
            발생한 운영 장애 내역이 없습니다.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left whitespace-nowrap">
              <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                <tr>
                  <th className="px-4 py-3 font-medium">관리번호</th>
                  <th className="px-4 py-3 font-medium">상태</th>
                  <th className="px-4 py-3 font-medium">심각도</th>
                  <th className="px-4 py-3 font-medium">모듈 / 엔드포인트</th>
                  <th className="px-4 py-3 font-medium">오류 메시지</th>
                  <th className="px-4 py-3 font-medium text-right">발생 횟수</th>
                  <th className="px-4 py-3 font-medium">담당자</th>
                  <th className="px-4 py-3 font-medium text-right">최근 발생</th>
                  <th className="px-4 py-3 font-medium"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {incidents.map((inc) => {
                  const SConf = STATUS_CONFIG[inc.status];
                  return (
                    <tr key={inc.id} className="hover:bg-slate-50/50 transition-colors">
                      <td className="px-4 py-3 font-medium text-slate-700">{inc.incidentNo}</td>
                      <td className="px-4 py-3">
                        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium border ${SConf.color}`}>
                          <SConf.icon className="size-3" />
                          {SConf.label}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${SEVERITY_COLOR[inc.severity]}`}>
                          {inc.severity}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex flex-col">
                          <span className="font-medium text-slate-800">{inc.moduleName}</span>
                          {inc.endpoint && <span className="text-xs text-slate-500 truncate max-w-[200px]">{inc.endpoint}</span>}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-slate-600 truncate inline-block max-w-[250px]" title={inc.errorMessage}>
                          {inc.errorMessage}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="inline-flex items-center gap-1 text-slate-600 font-medium bg-slate-100 px-2 rounded-full">
                          {inc.occurrenceCount}회
                        </span>
                      </td>
                      <td className="px-4 py-3 text-slate-600">
                        {inc.assignedToName || <span className="text-slate-400">미배정</span>}
                      </td>
                      <td className="px-4 py-3 text-right text-slate-500 text-xs">
                        {formatDateTime(inc.lastOccurredAt)}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <Button variant="ghost" size="sm" onClick={() => setSelectedIncidentId(inc.id)}>상세보기</Button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
