import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { incidentService, type Incident } from "@/services/incidentService";
import { formatDateTime } from "@/lib/format";
import jsPDF from "jspdf";
import html2canvas from "html2canvas-pro";

export function IncidentReportPrint() {
  const { id } = useParams();
  const [inc, setInc] = useState<Incident | null>(null);
  const [generating, setGenerating] = useState(false);
  const areaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (id) {
      incidentService.get(id).then(setInc).catch(console.error);
    }
    document.body.classList.add('print-mode');
    return () => document.body.classList.remove('print-mode');
  }, [id]);

  const handleDownloadPdf = async () => {
    const area = areaRef.current;
    if (!area || !inc) return;
    setGenerating(true);
    try {
      // 보고서 DOM 을 캔버스로 렌더 → A4 PDF 로 변환(여러 장 자동 분할).
      const canvas = await html2canvas(area, {
        scale: 2,
        useCORS: true,
        backgroundColor: "#ffffff",
      });
      const imgData = canvas.toDataURL("image/png");

      const pdf = new jsPDF({ unit: "mm", format: "a4", orientation: "portrait" });
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const imgWidth = pageWidth;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;

      let heightLeft = imgHeight;
      let position = 0;
      pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;
      while (heightLeft > 0) {
        position -= pageHeight;
        pdf.addPage();
        pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;
      }

      // 파일명은 관리번호 기반으로 동적 생성한다(고정 파일명 하드코딩 금지).
      pdf.save(`운영장애보고서_${inc.incidentNo}.pdf`);
    } catch (err) {
      console.error(err);
      alert("PDF 생성 중 오류가 발생했습니다. 다시 시도해 주세요.");
    } finally {
      setGenerating(false);
    }
  };

  if (!inc) return <div className="p-8">보고서 생성 중...</div>;

  return (
    <div className="bg-slate-100 min-h-screen py-6">
      <style dangerouslySetInnerHTML={{__html: `
        @media print {
          body * { visibility: hidden; }
          #print-area, #print-area * { visibility: visible; }
          #print-area { position: absolute; left: 0; top: 0; width: 100%; }
          .no-print { display: none; }
        }
      `}} />

      {/* 화면 전용 컨트롤 바 — PDF/인쇄 결과물에는 포함되지 않는다. */}
      <div className="no-print mx-auto mb-4 flex max-w-[210mm] items-center justify-end gap-2 px-2">
        <button
          type="button"
          onClick={handleDownloadPdf}
          disabled={generating}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {generating ? "PDF 생성 중..." : "PDF 다운로드"}
        </button>
        <button
          type="button"
          onClick={() => window.print()}
          className="rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50"
        >
          인쇄
        </button>
      </div>

      <div
        ref={areaRef}
        id="print-area"
        className="mx-auto min-h-[297mm] max-w-[210mm] bg-white p-8 text-black shadow"
      >
        <div className="text-center mb-10 pb-6 border-b-2 border-black">
          <h1 className="text-3xl font-bold tracking-tight mb-2">운영 장애(Incident) 조치 보고서</h1>
          <p className="text-sm text-gray-600">RENAI Medical AI System</p>
        </div>

        <table className="w-full border-collapse mb-8 text-sm">
          <tbody>
            <tr>
              <td className="border border-black bg-gray-100 p-2 font-bold w-32">관리번호</td>
              <td className="border border-black p-2 font-mono">{inc.incidentNo}</td>
              <td className="border border-black bg-gray-100 p-2 font-bold w-32">심각도</td>
              <td className="border border-black p-2 font-bold">{inc.severity}</td>
            </tr>
            <tr>
              <td className="border border-black bg-gray-100 p-2 font-bold">발생 모듈</td>
              <td className="border border-black p-2">{inc.moduleName}</td>
              <td className="border border-black bg-gray-100 p-2 font-bold">최초 발생일시</td>
              <td className="border border-black p-2">{formatDateTime(inc.firstOccurredAt)}</td>
            </tr>
            <tr>
              <td className="border border-black bg-gray-100 p-2 font-bold">엔드포인트</td>
              <td className="border border-black p-2">{inc.endpoint || '-'}</td>
              <td className="border border-black bg-gray-100 p-2 font-bold">최근 발생일시</td>
              <td className="border border-black p-2">{formatDateTime(inc.lastOccurredAt)}</td>
            </tr>
            <tr>
              <td className="border border-black bg-gray-100 p-2 font-bold">담당 조사관</td>
              <td className="border border-black p-2">{inc.assignedToName || '미지정'}</td>
              <td className="border border-black bg-gray-100 p-2 font-bold">발생 횟수</td>
              <td className="border border-black p-2">{inc.occurrenceCount}회</td>
            </tr>
          </tbody>
        </table>

        <div className="mb-8">
          <h2 className="text-lg font-bold border-b border-black mb-3 pb-1">1. 장애 개요 (오류 내용)</h2>
          <div className="p-3 border border-gray-300 min-h-[60px] whitespace-pre-wrap font-mono text-xs">
            {inc.errorMessage}
          </div>
        </div>

        <div className="mb-8">
          <h2 className="text-lg font-bold border-b border-black mb-3 pb-1">2. 원인 분석 (Root Cause)</h2>
          <div className="p-3 border border-gray-300 min-h-[100px] whitespace-pre-wrap text-sm">
            {inc.rootCause || '작성된 내용이 없습니다.'}
          </div>
        </div>

        <div className="mb-8">
          <h2 className="text-lg font-bold border-b border-black mb-3 pb-1">3. 조치 사항 및 재발 방지 (Action Taken)</h2>
          <div className="p-3 border border-gray-300 min-h-[150px] whitespace-pre-wrap text-sm">
            {inc.actionTaken || '작성된 내용이 없습니다.'}
          </div>
        </div>

        <div className="mt-16 flex justify-end">
          <table className="border-collapse text-sm">
            <tbody>
              <tr>
                <td rowSpan={2} className="border border-black bg-gray-100 p-2 font-bold text-center w-24">
                  최종 확인
                </td>
                <td className="border border-black p-2 text-center w-40">
                  {inc.resolvedByName || '서명 없음'}
                </td>
              </tr>
              <tr>
                <td className="border border-black h-24 relative p-2">
                  {inc.signaturePath && (
                    <img
                      src={"/api" + inc.signaturePath}
                      alt="Signature"
                      className="absolute inset-0 w-full h-full object-contain p-2"
                      crossOrigin="anonymous"
                    />
                  )}
                </td>
              </tr>
              <tr>
                <td colSpan={2} className="border border-black p-2 text-center text-xs text-gray-600">
                  {inc.lockedAt ? `LOCKED: ${formatDateTime(inc.lockedAt)}` : '미승인 상태'}
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="mt-10 text-center text-xs text-gray-500 pb-10">
          본 문서는 전자 서명되어 위변조가 불가능한 시스템 원본 기록의 사본입니다. <br/>
          발행 일시: {formatDateTime(new Date().toISOString())}
        </div>
      </div>
    </div>
  );
}
