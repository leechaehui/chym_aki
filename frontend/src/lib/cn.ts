import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * className 병합 유틸. clsx 로 조건부 클래스를 모으고 tailwind-merge 로 충돌을 해소한다.
 * shadcn/ui 컨벤션의 `cn` — 모든 UI 컴포넌트의 variant 합성에 사용한다.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
