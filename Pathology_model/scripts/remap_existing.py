"""
파일명 충돌 버그 복구 (1회성)

기존 파일은 '{pid}_{suffix}_{package_id[:8]}' 로 저장돼 같은 package의 여러 슬라이드가
서로 덮어써졌다. 새 네이밍은 '{pid}_{suffix}_{file_id[:8]}' (file_id=고유).

복구 전략(재다운로드 최소화):
 - old_name 이 매니페스트에서 '단일 행'에 대응하면(충돌 아님): 그 디스크 파일은
   그 행의 파일이 확실 -> new_name 으로 rename (살림).
 - old_name 이 '여러 행'에 대응하면(충돌): 디스크 파일이 어느 file_id인지 알 수 없음 ->
   삭제하고 해당 행들은 재다운로드.
"""
from pathlib import Path
import pandas as pd

BASE = (Path(__file__).resolve().parents[2] / "data/raw")
SEL = (Path(__file__).resolve().parents[2] / "selected_manifest.csv")


def old_name(r):
    pid = str(r["redcap_id"]).replace(";", "_"); suf = r["_grp"].replace("wsi_", "").upper()
    ext = ".tif" if r["_grp"] == "wsi_if" else ".svs"
    return f'{pid}_{suf}_{str(r["package_id"])[:8]}{ext}'


def new_name(r):
    pid = str(r["redcap_id"]).replace(";", "_"); suf = r["_grp"].replace("wsi_", "").upper()
    ext = ".tif" if r["_grp"] == "wsi_if" else ".svs"
    return f'{pid}_{suf}_{str(r["file_name"]).split("_")[0][:8]}{ext}'


def main():
    df = pd.read_csv(SEL)
    df["old"] = df.apply(old_name, axis=1)
    df["new"] = df.apply(new_name, axis=1)

    # file_id[:8] 고유성 검증 (그룹 내)
    dup = df.groupby("_grp")["new"].apply(lambda s: s.duplicated().sum()).sum()
    print(f"새 파일명 그룹내 중복: {dup}건 (0이어야 함)")
    if dup:
        print("경고: file_id[:8] 충돌 존재 -> 더 긴 접두 필요")
        return

    old_counts = df["old"].value_counts()
    renamed = deleted = redownload = saved_missing = 0

    for _, r in df.iterrows():
        grp_dir = BASE / r["_grp"]
        op = grp_dir / r["old"]
        np_ = grp_dir / r["new"]
        if old_counts[r["old"]] == 1:
            # 충돌 아님: 디스크의 old 파일을 new 로 rename
            if np_.exists():
                renamed += 1  # 이미 new 로 존재(이전 실행)
            elif op.exists():
                op.rename(np_)
                renamed += 1
            else:
                saved_missing += 1  # 아직 안 받음 -> 다운로더가 받을 것
        else:
            # 충돌: old 파일 삭제(어느 file_id인지 모름) -> 재다운로드
            if op.exists():
                op.unlink()
                deleted += 1
            redownload += 1

    # 매니페스트에 없는 잔여 old-style 파일 정리(혹시 남은 것)
    valid_new = set(df["new"])
    orphan = 0
    for g in df["_grp"].unique():
        d = BASE / g
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file() and f.suffix in (".svs", ".tif") and f.name not in valid_new:
                f.unlink(); orphan += 1

    print(f"rename(살림): {renamed}")
    print(f"충돌삭제: {deleted}  (해당 {redownload}행 재다운로드 예정)")
    print(f"고아파일 정리: {orphan}")
    # 현재 new 이름으로 디스크에 존재하는 수
    have = sum(1 for _, r in df.iterrows() if (BASE / r["_grp"] / r["new"]).exists())
    print(f"\n복구 후 디스크 보유: {have}/{len(df)}  (재다운로드 필요 {len(df)-have}개)")


if __name__ == "__main__":
    main()
