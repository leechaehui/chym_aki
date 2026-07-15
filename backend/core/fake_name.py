"""합성(가짜) 한글 표시명 생성 — 프론트 `lib/fakeName.ts` 의 백엔드 복제본.

MIMIC-IV 는 비식별화돼 실제 이름이 없다. subject_id 를 시드로 **결정적**으로 같은
이름을 만든다(같은 환자 = 항상 같은 이름). 프론트(fakeKoreanName)와 **바이트 단위로
동일한 결과**를 내야 이름 검색이 화면 표시명과 일치한다(알고리즘/배열을 그대로 옮김).

이름 풀(가족/음절1/음절2)은 30×40×40=48,000 조합 — 수백~수천 명 규모 코호트에서
서로 다른 환자가 같은 표시명을 받을 확률(생일 문제)을 낮게 유지하기 위해 20×20×20에서 확장함.
그래도 확률적 축소일 뿐 완전한 1:1 보장은 아니다(코호트가 아주 커지면 여전히 드물게 충돌 가능).
"""
from __future__ import annotations

FAMILY_NAMES = [
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
    "한", "오", "서", "신", "권", "황", "안", "송", "류", "전",
    "홍", "문", "손", "배", "백", "남", "유", "심", "노", "하",
]
GIVEN_SYLLABLES_1 = [
    "민", "서", "지", "현", "예", "준", "도", "하", "수", "은",
    "유", "주", "재", "성", "영", "정", "승", "윤", "다", "태",
    "아", "소", "나", "라", "우", "인", "혜", "경", "규", "훈",
    "상", "명", "근", "창", "병", "광", "덕", "익", "별", "홍",
]
GIVEN_SYLLABLES_2 = [
    "준", "우", "현", "진", "호", "빈", "아", "연", "원", "희",
    "수", "민", "서", "영", "건", "철", "경", "라", "은", "용",
    "재", "훈", "석", "규", "겸", "찬", "율", "오", "담", "결",
    "강", "웅", "범", "형", "욱", "록", "노", "화", "솔", "안",
]

_MASK = 0xFFFFFFFF

# 시나리오 주인공 등 특정 subject_id 는 데모 내러티브에 맞춰 표시명을 고정한다.
# (프론트 lib/fakeName.ts 의 PINNED_NAMES 와 반드시 동일하게 유지 — 화면/검색 일치.)
# 데모 트리거 버튼·가이드가 부르는 이름과 실제 목록 표시명이 어긋나지 않도록 함.
_PINNED_NAMES: dict[int, str] = {
    10218191: "오준현",   # 급성 악화 트리거 시나리오 주인공
}


def _imul(a: int, b: int) -> int:
    """JS Math.imul 등가 — 32bit 곱셈(하위 32비트)."""
    return ((a & _MASK) * (b & _MASK)) & _MASK


def _hash_seed(seed: int) -> int:
    """JS hashSeed 등가 — FNV offset XOR 후 2단 혼합(부호없는 32bit)."""
    h = (2166136261 ^ (seed & _MASK)) & _MASK
    h = _imul(h ^ (h >> 15), 2246822507)
    h = _imul(h ^ (h >> 13), 3266489909)
    return (h ^ (h >> 16)) & _MASK


def fake_korean_name(subject_id: int) -> str:
    """subject_id → 결정적 한글 표시명(예: 30037325 → '권영용').

    _PINNED_NAMES 에 등록된 subject_id 는 해시 대신 고정 표시명을 쓴다."""
    pinned = _PINNED_NAMES.get(int(subject_id))
    if pinned is not None:
        return pinned
    h = _hash_seed(int(subject_id))
    return (
        FAMILY_NAMES[h % len(FAMILY_NAMES)]
        + GIVEN_SYLLABLES_1[(h >> 8) % len(GIVEN_SYLLABLES_1)]
        + GIVEN_SYLLABLES_2[(h >> 16) % len(GIVEN_SYLLABLES_2)]
    )
