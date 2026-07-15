import { useEffect, useState, useMemo } from "react";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, PieChart, Pie, Cell 
} from "recharts";
import { Activity, AlertTriangle, Clock, Database, Zap, ShieldAlert, Cpu, X } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { api } from "@/services/http";

export function TelemetryAdmin() {
  const navigate = useNavigate();
  const [stats, setStats] = useState<any>(null);
  const [chartData, setChartData] = useState<any[]>([]);
  const [logs, setLogs] = useState<any[]>([]);
  const [activeReqs, setActiveReqs] = useState<any[]>([]);
  const [chartDate, setChartDate] = useState<string>(""); // empty string means today
  const [currentPage, setCurrentPage] = useState(1);

  const errorStats = useMemo(() => {
    if (!stats || !stats.error_distribution || stats.error_distribution.length === 0) return [];
    
    const dist = stats.error_distribution;
    const totalErrors = dist.reduce((sum: number, item: any) => sum + item.count, 0);
    if (totalErrors === 0) return [];

    const colors = ['#f43f5e', '#f59e0b', '#8b5cf6', '#ec4899', '#f97316'];
    
    return dist.map((item: any, i: number) => ({
      name: `HTTP ${item.status_code}`,
      value: item.count,
      percent: Math.round((item.count / totalErrors) * 100),
      color: item.status_code >= 500 ? '#f43f5e' : (colors[i % colors.length])
    }));
  }, [stats]);

  useEffect(() => {
    const fetchData = () => {
      const params = chartDate ? `?date=${chartDate}` : "";

      // Fetch real data from the backend
      api.get(`/admin/telemetry/stats${params}`)
        .then((res: any) => setStats(res))
        .catch((err) => {
          console.error("Failed to load telemetry stats:", err);
          setStats((prev: any) => prev || { total_requests: 0, error_rate: 0, avg_latency_ms: 0, dropped_logs: 0, memory_usage: 0 });
        });

      api.get(`/admin/telemetry/chart${params}`)
        .then((res: any) => setChartData(res))
        .catch((err) => {
          console.error("Failed to load telemetry chart data:", err);
          setChartData((prev) => prev || []);
        });

      api.get(`/admin/telemetry/recent${params}`)
        .then((res: any) => setLogs(res))
        .catch((err) => {
          console.error("Failed to load recent logs:", err);
          setLogs((prev) => prev || []);
        });

      api.get("/admin/telemetry/active")
        .then((res: any) => setActiveReqs(res))
        .catch((err) => {
          console.error("Failed to load active requests:", err);
          setActiveReqs([]);
        });
    };

    setCurrentPage(1); // Reset page on date change
    fetchData(); // Initial fetch
    const intervalId = setInterval(fetchData, 3000); // Poll every 3 seconds for real-time feel

    return () => clearInterval(intervalId);
  }, [chartDate]);

  const itemsPerPage = 5;
  const totalPages = Math.ceil(logs.length / itemsPerPage) || 1;
  const paginatedLogs = logs.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  if (!stats) return <div className="p-10 text-white flex items-center gap-2"><Activity className="animate-spin" /> 서버 모니터링 로딩 중...</div>;

  return (
    <div className="relative min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10 font-sans selection:bg-indigo-500/30">
      
      <button onClick={() => navigate(-1)} className="absolute top-4 right-4 md:top-8 md:right-8 p-2 rounded-full bg-slate-900/50 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors border border-slate-800 shadow-lg z-50" title="닫기">
        <X size={24} />
      </button>

      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-10 gap-4 pr-12 md:pr-16">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-cyan-400 flex items-center gap-3 tracking-tight">
            <Zap className="text-indigo-400" size={36} />
            서버 통합 모니터링
          </h1>
          <p className="text-slate-400 mt-2 text-sm md:text-base max-w-2xl">
            전체 시스템의 실시간 트래픽, API 응답 시간, 오류 발생 현황 및 메모리 사용 상태를 한눈에 파악할 수 있습니다.
          </p>
        </div>
        <div className="flex items-center gap-3 bg-slate-900/50 backdrop-blur-md border border-slate-800 px-4 py-2 rounded-full shadow-lg">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_10px_rgba(52,211,153,0.8)]" />
          <span className="text-sm font-medium tracking-wide">시스템 정상 작동 중</span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        <KpiCard icon={<Activity className="text-cyan-400" />} title="총 누적 요청 수 (24시간)" value={stats.total_requests.toLocaleString()} trend="+12.5%" />
        <KpiCard icon={<Clock className="text-indigo-400" />} title="평균 응답 시간" value={`${stats.avg_latency_ms} ms`} trend="-2.1 ms" trendGood={true} />
        <KpiCard icon={<AlertTriangle className="text-rose-400" />} title="오류 발생률" value={`${stats.error_rate}%`} trend="+0.05%" trendGood={false} />
        <KpiCard icon={<Cpu className="text-amber-400" />} title="서버 메모리 사용량" value={`${stats.memory_usage}%`} subtitle={`과부하 방지로 ${stats.dropped_logs}개 기록 생략`} />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-10">
        <div className="col-span-1 lg:col-span-2 bg-slate-900/40 backdrop-blur-xl border border-slate-800/60 rounded-3xl p-6 shadow-2xl relative overflow-hidden group flex flex-col">
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />
          <div className="flex flex-wrap justify-between items-center mb-6 gap-4 relative z-10">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Activity size={20} className="text-indigo-400" /> 시간대별 트래픽 및 응답 시간
            </h2>
            <div className="flex bg-slate-800/80 rounded-lg border border-slate-700 relative">
              <select 
                value={chartDate} 
                onChange={(e) => setChartDate(e.target.value)}
                className="bg-transparent text-slate-300 text-sm font-medium focus:outline-none appearance-none pr-8 pl-4 py-2 cursor-pointer hover:text-white"
              >
                {[0, 1, 2, 3, 4, 5, 6].map(days => {
                  const d = new Date(Date.now() - days * 86400000);
                  const dateStr = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
                  // value="" for today (days === 0) so the API uses the default "today" logic
                  const val = days === 0 ? "" : dateStr;
                  return <option key={days} value={val} className="bg-slate-800">{dateStr}</option>;
                })}
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-400">
                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/></svg>
              </div>
            </div>
          </div>
          <div className="flex-1 w-full relative z-10 min-h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorReq" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#818cf8" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis 
                  dataKey="time" 
                  stroke="#94a3b8" 
                  fontSize={12} 
                  tickLine={false} 
                  axisLine={false} 
                  interval={0}
                  tickFormatter={(val) => {
                    const hour = parseInt(val.split(":")[0], 10);
                    return `${hour}시`;
                  }}
                />
                <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `${val.toLocaleString()}건`} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderRadius: '12px', border: '1px solid #334155', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)' }}
                  itemStyle={{ color: '#e2e8f0' }}
                />
                <Area type="monotone" dataKey="requests" stroke="#818cf8" strokeWidth={3} fillOpacity={1} fill="url(#colorReq)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="col-span-1 bg-slate-900/40 backdrop-blur-xl border border-slate-800/60 rounded-3xl p-6 shadow-2xl">
          <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <ShieldAlert size={20} className="text-rose-400" /> 오류 유형 분포
          </h2>
          <div className="flex flex-col justify-center items-center h-80">
            {errorStats.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-emerald-400">
                <div className="w-32 h-32 rounded-full border-8 border-emerald-500/20 flex items-center justify-center mb-4 relative">
                   <div className="absolute inset-0 border-8 border-emerald-500 rounded-full" />
                   <span className="text-2xl font-bold">0%</span>
                </div>
                <span className="text-sm font-medium">최근 오류 없음</span>
              </div>
            ) : (
              <>
                <div className="w-full h-48 relative flex justify-center items-center">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={errorStats}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="value"
                        stroke="none"
                      >
                        {errorStats.map((entry: any, index: number) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip 
                        formatter={(value, name, props) => [`${value}건 (${props.payload.percent}%)`, name]}
                        contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderRadius: '8px', border: '1px solid #334155' }}
                        itemStyle={{ color: '#e2e8f0' }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                    <div className="text-2xl font-bold text-white">{errorStats[0]?.percent}%</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">{errorStats[0]?.name}</div>
                  </div>
                </div>
                <div className="mt-6 w-full space-y-3 px-4 max-h-28 overflow-y-auto custom-scrollbar">
                  {errorStats.map((stat: any, i: number) => (
                    <div key={i} className="flex justify-between text-sm items-center">
                      <span className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full shadow-sm" style={{ backgroundColor: stat.color }} /> 
                        {stat.name}
                      </span>
                      <span className="font-mono text-slate-300">{stat.percent}% <span className="text-slate-500 text-xs ml-1">({stat.value}건)</span></span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Active Requests */}
      {activeReqs.length > 0 && (
        <div className="bg-slate-900/40 backdrop-blur-xl border border-indigo-500/50 rounded-3xl p-6 shadow-2xl overflow-hidden mb-10">
          <h2 className="text-lg font-semibold mb-6 flex items-center gap-2 text-indigo-400">
            <Activity size={20} className="animate-pulse" /> 현재 진행 중인 요청 <span className="bg-indigo-500/20 text-indigo-300 text-xs px-2 py-0.5 rounded-full">{activeReqs.length}건</span>
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 text-xs uppercase tracking-wider">
                  <th className="pb-4 font-semibold px-4">요청 ID</th>
                  <th className="pb-4 font-semibold px-4">메서드 / 엔드포인트</th>
                  <th className="pb-4 font-semibold px-4">접속 IP</th>
                  <th className="pb-4 font-semibold px-4">시작 시간</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                {activeReqs.map((req, i) => (
                  <tr key={i} className="border-b border-slate-800/50 bg-indigo-500/5 hover:bg-indigo-500/10 transition-colors">
                    <td className="py-4 px-4 font-mono text-xs text-slate-400">{req.request_id}</td>
                    <td className="py-4 px-4">
                      <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold mr-2 ${req.method === 'GET' ? 'bg-blue-500/20 text-blue-400' : req.method === 'POST' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>{req.method}</span>
                      <span className="text-slate-300">{req.endpoint}</span>
                    </td>
                    <td className="py-4 px-4 text-slate-400">{req.ip_address || '-'}</td>
                    <td className="py-4 px-4 text-slate-400">{req.start_time}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recent Logs Table */}
      <div className="bg-slate-900/40 backdrop-blur-xl border border-slate-800/60 rounded-3xl p-6 shadow-2xl overflow-hidden">
        <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
          <Database size={20} className="text-emerald-400" /> 실시간 서버 접속 및 API 로그
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-xs uppercase tracking-wider">
                <th className="pb-4 font-semibold px-4">요청 ID</th>
                <th className="pb-4 font-semibold px-4">메서드 / 엔드포인트</th>
                <th className="pb-4 font-semibold px-4">상태 코드</th>
                <th className="pb-4 font-semibold px-4">응답 시간 (ms)</th>
                <th className="pb-4 font-semibold px-4">발생 시간</th>
                <th className="pb-4 font-semibold px-4">샘플링 여부</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              {paginatedLogs.map((log, i) => (
                <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                  <td className="py-4 px-4 font-mono text-xs text-slate-400">{log.request_id}</td>
                  <td className="py-4 px-4">
                    <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold mr-2 ${log.method === 'GET' ? 'bg-blue-500/20 text-blue-400' : log.method === 'POST' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>{log.method}</span>
                    <span className="text-slate-300">{log.endpoint}</span>
                  </td>
                  <td className="py-4 px-4">
                    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${log.status_code >= 500 ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' : log.status_code >= 400 ? 'bg-amber-500/20 text-amber-400' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'}`}>
                      {log.status_code >= 500 && <AlertTriangle size={12} />}
                      {log.status_code}
                    </span>
                  </td>
                  <td className="py-4 px-4 font-mono text-slate-300">{log.latency_ms} ms</td>
                  <td className="py-4 px-4 text-slate-400">{log.time}</td>
                  <td className="py-4 px-4">
                    {log.is_sampled ? (
                      <span className="px-2 py-1 bg-indigo-500/20 text-indigo-300 text-xs rounded border border-indigo-500/30">샘플링됨</span>
                    ) : (
                      <span className="text-slate-500 text-xs">-</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {/* Pagination Controls */}
        {logs.length > 0 && (
          <div className="flex flex-col items-center justify-center mt-6 px-4 space-y-3">
            <div className="flex items-center gap-1">
              {/* 맨 처음으로 (<<) */}
              <button
                onClick={() => setCurrentPage(1)}
                disabled={currentPage === 1}
                className="w-8 h-8 flex items-center justify-center text-xs font-bold rounded-md text-slate-400 hover:bg-slate-800 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                title="첫 페이지"
              >
                &laquo;
              </button>
              
              {/* 이전 (<) */}
              <button
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="w-8 h-8 flex items-center justify-center text-xs font-bold rounded-md text-slate-400 hover:bg-slate-800 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                title="이전 페이지"
              >
                &lsaquo;
              </button>
              
              {/* 페이지 번호 (최대 5개 표시) */}
              <div className="flex items-center gap-1 mx-2">
                {Array.from({ length: totalPages }).map((_, idx) => {
                  const pageNum = idx + 1;
                  // 현재 페이지 기준으로 앞뒤 2개씩 표시 (총 5개)
                  let startPage = Math.max(1, currentPage - 2);
                  let endPage = Math.min(totalPages, startPage + 4);
                  if (endPage - startPage < 4) {
                    startPage = Math.max(1, endPage - 4);
                  }
                  
                  if (pageNum >= startPage && pageNum <= endPage) {
                    return (
                      <button
                        key={pageNum}
                        onClick={() => setCurrentPage(pageNum)}
                        className={`w-8 h-8 flex items-center justify-center text-xs font-medium rounded-md transition-colors ${currentPage === pageNum ? 'bg-indigo-500 text-white shadow-md' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'}`}
                      >
                        {pageNum}
                      </button>
                    );
                  }
                  return null;
                })}
              </div>
              
              {/* 다음 (>) */}
              <button
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="w-8 h-8 flex items-center justify-center text-xs font-bold rounded-md text-slate-400 hover:bg-slate-800 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                title="다음 페이지"
              >
                &rsaquo;
              </button>
              
              {/* 맨 끝으로 (>>) */}
              <button
                onClick={() => setCurrentPage(totalPages)}
                disabled={currentPage === totalPages}
                className="w-8 h-8 flex items-center justify-center text-xs font-bold rounded-md text-slate-400 hover:bg-slate-800 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                title="마지막 페이지"
              >
                &raquo;
              </button>
            </div>
            
            <span className="text-[10px] text-slate-500">
              총 {logs.length}개의 로그 ({currentPage}/{totalPages} 페이지)
            </span>
          </div>
        )}
      </div>

    </div>
  );
}

function KpiCard({ icon, title, value, trend, trendGood, subtitle }: any) {
  return (
    <div className="bg-slate-900/40 backdrop-blur-xl border border-slate-800/60 rounded-3xl p-6 shadow-lg hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 group">
      <div className="flex justify-between items-start mb-4">
        <div className="p-3 rounded-2xl bg-slate-800/80 shadow-inner group-hover:scale-110 transition-transform">
          {icon}
        </div>
        {trend && (
          <span className={`text-xs font-bold px-2 py-1 rounded-full ${trendGood !== false ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>
            {trend}
          </span>
        )}
      </div>
      <div>
        <h3 className="text-slate-400 text-sm font-medium">{title}</h3>
        <div className="text-2xl md:text-3xl font-bold mt-1 tracking-tight text-white">{value}</div>
        {subtitle && <p className="text-xs text-slate-500 mt-2">{subtitle}</p>}
      </div>
    </div>
  );
}
