import { useState, useRef } from "react";
import { useAuthStore } from "@/store/authStore";
import { api } from "@/services/http";

export function ProfilePage() {
  const user = useAuthStore((s) => s.user);
  const setUser = useAuthStore((s) => s.setUser);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    
    // Get mouse/touch coordinates relative to the canvas
    const rect = canvas.getBoundingClientRect();
    // 캔버스 버퍼 크기와 화면 표시 크기가 다르므로 좌표를 버퍼 기준으로 환산.
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (("touches" in e ? e.touches[0].clientX : (e as React.MouseEvent).clientX) - rect.left) * scaleX;
    const y = (("touches" in e ? e.touches[0].clientY : (e as React.MouseEvent).clientY) - rect.top) * scaleY;
    
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    
    const rect = canvas.getBoundingClientRect();
    // 캔버스 버퍼 크기와 화면 표시 크기가 다르므로 좌표를 버퍼 기준으로 환산.
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (("touches" in e ? e.touches[0].clientX : (e as React.MouseEvent).clientX) - rect.left) * scaleX;
    const y = (("touches" in e ? e.touches[0].clientY : (e as React.MouseEvent).clientY) - rect.top) * scaleY;
    
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  const handleSaveSignature = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    setLoading(true);
    setMessage("");
    try {
      const dataUrl = canvas.toDataURL("image/png");
      const updatedUser = await api.post("/auth/me/signature", { signatureBase64: dataUrl });
      setUser(updatedUser as any);
      setMessage("서명이 성공적으로 저장되었습니다.");
    } catch (err: any) {
      setMessage("서명 저장에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  };

  if (!user) return null;

  return (
    <div className="p-8 max-w-4xl mx-auto h-full overflow-y-auto">
      <h1 className="text-3xl font-light mb-8">내 프로필</h1>
      
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mb-8">
        <h2 className="text-xl font-medium mb-4">계정 정보</h2>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-slate-500 block">이름</span>
            <span className="font-medium text-slate-800">{user.name}</span>
          </div>
          <div>
            <span className="text-slate-500 block">아이디</span>
            <span className="font-medium text-slate-800">{user.username}</span>
          </div>
          <div>
            <span className="text-slate-500 block">부서</span>
            <span className="font-medium text-slate-800">{user.department}</span>
          </div>
          <div>
            <span className="text-slate-500 block">권한</span>
            <span className="font-medium text-slate-800">{user.role}</span>
          </div>
        </div>
      </div>
      
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-xl font-medium mb-4">전자서명 등록</h2>
        <p className="text-sm text-slate-500 mb-4">장애 조치(Incident) 등 주요 승인 업무에 사용할 서명을 마우스나 터치로 그려주세요.</p>
        
        {user.signaturePath && (
          <div className="mb-6 p-4 border border-blue-100 bg-blue-50/50 rounded-lg">
            <p className="text-sm text-blue-800 font-medium mb-2">현재 등록된 서명</p>
            <img src={"/api" + user.signaturePath} alt="My Signature" className="h-16 border border-slate-200 bg-white" />
          </div>
        )}
        
        <div className="border-2 border-dashed border-slate-300 rounded-lg overflow-hidden bg-slate-50 w-full max-w-md mx-auto mb-4">
          <canvas
            ref={canvasRef}
            width={400}
            height={200}
            className="w-full h-full cursor-crosshair touch-none bg-white"
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseOut={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
          />
        </div>
        
        <div className="flex justify-center items-center gap-4">
          <button 
            onClick={clearCanvas}
            className="px-4 py-2 border border-slate-300 rounded-md text-sm font-medium hover:bg-slate-50"
            disabled={loading}
          >
            지우기
          </button>
          <button 
            onClick={handleSaveSignature}
            className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
            disabled={loading}
          >
            {loading ? "저장 중..." : "서명 저장"}
          </button>
        </div>
        
        {message && (
          <p className="mt-4 text-center text-sm font-medium text-blue-600">
            {message}
          </p>
        )}
      </div>
    </div>
  );
}
