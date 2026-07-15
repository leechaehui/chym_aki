/**
 * 브라우저 내장 Web Audio API를 활용하여 단순한 비프음을 재생합니다.
 * 외부 오디오 파일 없이 경고음을 발생시킵니다.
 */

// 오디오 컨텍스트는 브라우저 정책(사용자 상호작용 후 활성화) 때문에 
// 필요할 때 지연 생성(Lazy initialization)합니다.
let audioCtx: AudioContext | null = null;

export function playBeep() {
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    
    // 만약 Suspended 상태라면 재개 시도 (브라우저 정책)
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }

    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    // 급박한 알람 형태의 주파수 설정 (예: 880Hz, A5 음)
    oscillator.type = 'square';
    oscillator.frequency.setValueAtTime(880, audioCtx.currentTime);
    
    // 짧고 강렬한 비프음 (볼륨 페이드아웃 적용)
    gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime); // 볼륨 낮춤 (너무 시끄럽지 않게)
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.3);

    oscillator.start(audioCtx.currentTime);
    oscillator.stop(audioCtx.currentTime + 0.3);
  } catch (err) {
    console.warn("Audio play failed (browser policy or not supported):", err);
  }
}
