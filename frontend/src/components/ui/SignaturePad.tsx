import React, { useRef, useState, useEffect } from "react";
import { Button } from "./button";
import { Eraser } from "lucide-react";

interface SignaturePadProps {
  onSign: (signatureBase64: string | null) => void;
  className?: string;
}

export function SignaturePad({ onSign, className = "" }: SignaturePadProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasSignature, setHasSignature] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set background to white
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Set drawing style
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#000000";
  }, []);

  const getCoordinates = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();

    // 캔버스 내부 버퍼(width/height)와 화면 표시 크기(rect)가 다르므로
    // 표시 좌표를 버퍼 좌표로 환산해야 마우스 위치에 정확히 그려진다.
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // Check if it's a touch event
    if ('touches' in e) {
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY
      };
    }

    // It's a mouse event
    return {
      x: ((e as React.MouseEvent).clientX - rect.left) * scaleX,
      y: ((e as React.MouseEvent).clientY - rect.top) * scaleY
    };
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault(); // Prevent scrolling on touch
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    
    setIsDrawing(true);
    const { x, y } = getCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault(); // Prevent scrolling on touch
    if (!isDrawing) return;
    
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    
    const { x, y } = getCoordinates(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    setHasSignature(true);
  };

  const stopDrawing = () => {
    if (isDrawing) {
      setIsDrawing(false);
      updateSignature();
    }
  };

  const clearSignature = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setHasSignature(false);
    onSign(null);
  };

  const updateSignature = () => {
    if (!hasSignature) {
      onSign(null);
      return;
    }
    const canvas = canvasRef.current;
    if (canvas) {
      onSign(canvas.toDataURL("image/png"));
    }
  };

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <div className="relative border-2 border-slate-300 rounded-md overflow-hidden bg-white">
        <canvas
          ref={canvasRef}
          width={400}
          height={150}
          className="w-full h-[150px] touch-none cursor-crosshair"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
        />
        {!hasSignature && (
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center text-slate-300 font-medium">
            여기에 서명해 주세요
          </div>
        )}
      </div>
      <div className="flex justify-end">
        <Button 
          type="button" 
          variant="outline" 
          size="sm" 
          onClick={clearSignature}
          disabled={!hasSignature}
          className="text-xs h-7"
        >
          <Eraser className="size-3.5 mr-1" /> 다시 그리기
        </Button>
      </div>
    </div>
  );
}
