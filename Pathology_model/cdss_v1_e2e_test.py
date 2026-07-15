import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json

from mil.cdss_v1_model import CDSSDeploymentModel
from mil.cdss_v1_cbr import CBRRetrievalEngine
from mil.cdss_v1_engine import ClinicalReferenceAggregator

ROOT = Path("D:/cdss_core")

def main():
    print("--- [MFDS SaMD v1.1 End-to-End Test] ---")
    
    # 1. Initialize Components
    model = CDSSDeploymentModel()
    model.eval()
    
    cbr = CBRRetrievalEngine()
    engine = ClinicalReferenceAggregator(config_path="mil/cdss_config.json")
    
    # 2. Load Manifest to build fake DB
    try:
        sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    except:
        print("Manifest not found, creating dummy DB.")
        sm = pd.DataFrame([{"patient_id": f"P-{i}", "kdigo_stage": 1, "task_ati_severity": 1} for i in range(70)])
    
    print("Building CBR Database...")
    meta_list = []
    z_concat_list = []
    
    with torch.no_grad():
        for _, row in sm.iterrows():
            # Create a fake input embedding for the patient just to populate DB
            dummy_h = torch.randn(1, 768)
            out = model(dummy_h)
            
            meta_list.append({
                "patient_id": row["patient_id"],
                "z_ais": out["z_ais"],
                "z_cds": out["z_cds"],
                "z_ins": out["z_ins"],
                "clinical_outcome": f"KDIGO Stage {row['kdigo_stage']} | ATI: {row['task_ati_severity']}"
            })
            z_concat_list.append(out["z_concat"])
            
    cbr.build_index(z_concat_list, meta_list)
    
    # 3. Simulate specific patient (30-10123)
    target_patient = "30-10123"
    print(f"\nSimulating inference for patient: {target_patient}")
    cbr.current_patient = target_patient
    
    dummy_input = torch.randn(10, 768) # 10 patches
    with torch.no_grad():
        out = model(dummy_input)
    
    # 4. CBR Retrieval
    cbr_results = cbr.retrieve(out["z_concat"], out["z_ais"], out["z_cds"], out["z_ins"], k=3)
    
    # 5. Engine Processing
    final_json = engine.process_patient(target_patient, out, cbr_results, input_tensor=dummy_input)
    
    # Save and print
    json_path = "C:/Users/301-4/.gemini/antigravity-ide/brain/2b055707-f32c-4739-8475-071c95614018/artifacts/samd_v1.1_output.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)
        
    print(f"\nSaved SaMD output to {json_path}")
    print(json.dumps(final_json, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
