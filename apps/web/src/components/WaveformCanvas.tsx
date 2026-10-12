import { useRef, useEffect, useCallback, type CanvasHTMLAttributes, forwardRef } from "react";

export interface WaveformCanvasProps extends CanvasHTMLAttributes<HTMLCanvasElement> {
  audioData: Float32Array | null;
  coughMarkers?: number[];
  sampleRate?: number;
  maxDurationMs?: number;
}

export const WaveformCanvas = forwardRef<HTMLCanvasElement, WaveformCanvasProps>(
  ({ audioData, coughMarkers = [], sampleRate = 16000, maxDurationMs = 30000, className = "", ...props }, _ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationRef = useRef<number>();

    const draw = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      const width = rect.width;
      const height = rect.height;
      const centerY = height / 2;
      const maxAmplitude = height * 0.4;

      ctx.clearRect(0, 0, width, height);

      if (!audioData || audioData.length === 0) {
        ctx.strokeStyle = "#e5e7eb";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.stroke();
        return;
      }

      const samplesPerPixel = Math.max(1, audioData.length / width);
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 1.5;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      ctx.beginPath();
      for (let x = 0; x < width; x++) {
        const start = Math.floor(x * samplesPerPixel);
        const end = Math.min(Math.floor((x + 1) * samplesPerPixel), audioData.length);
        let min = 0;
        let max = 0;
        for (let i = start; i < end; i++) {
          const v = audioData[i];
          if (v < min) min = v;
          if (v > max) max = v;
        }
        const y1 = centerY - max * maxAmplitude;
        const y2 = centerY - min * maxAmplitude;
        ctx.moveTo(x, y1);
        ctx.lineTo(x, y2);
      }
      ctx.stroke();

      if (coughMarkers.length > 0) {
        const maxSamples = (maxDurationMs / 1000) * sampleRate;
        ctx.fillStyle = "#ef4444";
        coughMarkers.forEach((sampleIndex) => {
          if (sampleIndex < maxSamples) {
            const x = (sampleIndex / maxSamples) * width;
            ctx.beginPath();
            ctx.arc(x, centerY, 4, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      }
    }, [audioData, coughMarkers, sampleRate, maxDurationMs]);

    useEffect(() => {
      draw();
      animationRef.current = requestAnimationFrame(() => {});
      return () => {
        if (animationRef.current) cancelAnimationFrame(animationRef.current);
      };
    }, [audioData, coughMarkers, draw]);

    useEffect(() => {
      const handleResize = () => draw();
      window.addEventListener("resize", handleResize);
      return () => window.removeEventListener("resize", handleResize);
    }, [draw]);

    return <canvas ref={canvasRef} className={`w-full h-32 ${className}`} {...props} />;
  }
);

WaveformCanvas.displayName = "WaveformCanvas";