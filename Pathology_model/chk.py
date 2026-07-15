import pandas as pd
unproc=set(l.strip() for l in open("/tmp/unproc.txt") if l.strip())
o=open("/tmp/chk.txt","w",encoding="utf-8")
def w(*a): print(*a,file=o)
for f,key in [("artifacts/split_manifest.csv","patient_id"),("artifacts/selected_manifest.csv","redcap_id"),("C:/team/chym_aki/split_manifest.csv","patient_id")]:
    try:
        d=pd.read_csv(f,dtype=str)
        ids=set(d[key].astype(str).str.strip().unique())
        w(f"{f}: 환자 {len(ids)}, 31미처리중 포함 {len(unproc & ids)}")
    except Exception as e: w(f,"ERR",e)
w("미처리 31명 중 split(C)에 있는:", sorted(unproc & set(pd.read_csv('C:/team/chym_aki/split_manifest.csv',dtype=str).patient_id.astype(str)))[:40])
o.close()
