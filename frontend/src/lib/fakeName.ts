/**
 * 합성(가짜) 한글 표시명 생성 — MIMIC-IV 는 비식별화돼 실제 이름이 없다.
 *
 * subjectId 를 시드로 **결정적**으로 같은 이름을 만든다(같은 환자 = 항상 같은 이름).
 * 임상 판단용 식별자가 아니라 화면 가독성을 위한 표시명일 뿐이며, 화면에는 반드시
 * "합성 표시명" 고지를 함께 노출한다(SYNTHETIC_NAME_NOTICE).
 */

const FAMILY_NAMES = [
  "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
  "한", "오", "서", "신", "권", "황", "안", "송", "류", "전",
  "홍", "문", "손", "배", "백", "남", "유", "심", "노", "하",
];

const GIVEN_SYLLABLES_1 = [
  "민", "서", "지", "현", "예", "준", "도", "하", "수", "은",
  "유", "주", "재", "성", "영", "정", "승", "윤", "다", "태",
  "아", "소", "나", "라", "우", "인", "혜", "경", "규", "훈",
  "상", "명", "근", "창", "병", "광", "덕", "익", "별", "홍",
];

const GIVEN_SYLLABLES_2 = [
  "준", "우", "현", "진", "호", "빈", "아", "연", "원", "희",
  "수", "민", "서", "영", "건", "철", "경", "라", "은", "용",
  "재", "훈", "석", "규", "겸", "찬", "율", "오", "담", "결",
  "강", "웅", "범", "형", "욱", "록", "노", "화", "솔", "안",
];

/**
 * 시나리오 주인공 등 특정 subjectId 는 데모 내러티브에 맞춰 표시명을 고정한다.
 * (백엔드 core/fake_name.py 의 _PINNED_NAMES 와 반드시 동일하게 유지 — 화면/검색 일치.)
 */
const PINNED_NAMES: Record<number, string> = {
  10218191: "오준현", // 급성 악화 트리거 시나리오 주인공
};

/** 문자열/숫자 시드 → 32bit 정수(결정적). */
function hashSeed(seed: number): number {
  let h = 2166136261 ^ seed;
  h = Math.imul(h ^ (h >>> 15), 2246822507);
  h = Math.imul(h ^ (h >>> 13), 3266489909);
  return (h ^ (h >>> 16)) >>> 0;
}

/** subjectId → 결정적 한글 가짜 이름(예: "김민준"). */
export function fakeKoreanName(subjectId: number): string {
  const pinned = PINNED_NAMES[subjectId];
  if (pinned !== undefined) return pinned;
  const h = hashSeed(subjectId);
  const family = FAMILY_NAMES[h % FAMILY_NAMES.length];
  const first = GIVEN_SYLLABLES_1[(h >>> 8) % GIVEN_SYLLABLES_1.length];
  const second = GIVEN_SYLLABLES_2[(h >>> 16) % GIVEN_SYLLABLES_2.length];
  return `${family}${first}${second}`;
}

/** 화면 고지 문구 — 이 화면의 환자/이름은 비식별 합성 데이터임을 분명히 한다. */
export const SYNTHETIC_NAME_NOTICE =
  "표시된 환자 이름은 비식별 MIMIC-IV 데이터에 부여한 합성(가짜) 표시명입니다. 실제 환자가 아닙니다.";
