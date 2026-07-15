import { useEffect, useState } from "react";
import { getToken } from "@/services/http";

/**
 * 인증이 필요한 이미지 URL을 fetch로 받아 blob URL로 변환한다.
 * <img src>는 커스텀 헤더(Authorization)를 실을 수 없어, 토큰이 필요한 리소스는
 * 직접 fetch한 뒤 blob URL을 만들어 <img>에 넣어야 한다.
 */
export function useAuthenticatedImage(url: string | null): string | null {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    if (!url) {
      setSrc(null);
      return;
    }

    let cancelled = false;
    let objectUrl: string | null = null;
    const token = getToken();

    fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
      .then((res) => {
        if (!res.ok) throw new Error(`image fetch failed: ${res.status}`);
        return res.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setSrc(objectUrl);
      })
      .catch(() => {
        if (!cancelled) setSrc(null);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url]);

  return src;
}
