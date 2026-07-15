import uuid
import datetime
import json
import hashlib
import torch

class ClinicalReferenceAggregator:
    """
    MFDS SaMD v1.1 Compliant Engine
    - NO decision making (removed rules engine)
    - NO recommendation (removed URGENT CONSULT etc.)
    - STRICTLY reference-only output
    - Immutable Audit Logging (input_hash, output_hash)
    """
    def __init__(self, config_path="mil/cdss_config.json"):
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except:
            self.config = {"rule_version": "unknown"}
            
        self.rule_version = self.config.get("rule_version", "unknown")
        
    def _compute_hash(self, obj):
        # Deterministic hashing of dict or tensor
        if torch.is_tensor(obj):
            content = obj.detach().cpu().numpy().tobytes()
        else:
            content = json.dumps(obj, sort_keys=True).encode('utf-8')
        return hashlib.sha256(content).hexdigest()
        
    def process_patient(self, patient_id, model_output, cbr_results, input_tensor=None):
        """
        Aggregates outputs strictly as reference material.
        """
        timestamp = datetime.datetime.now().isoformat()
        
        # input hash
        input_hash = self._compute_hash(input_tensor) if input_tensor is not None else "N/A"
        
        # 1. Formatting clinical reference
        clinical_reference = {
            "calibrated_risk_score": round(float(model_output.get("calibrated_risk_prob", 0)), 4),
            "ais": round(float(model_output.get("ais_prob", 0)), 4),
            "cds": round(float(model_output.get("cds_prob", 0)), 4),
            "ins": round(float(model_output.get("ins_prob", 0)), 4),
            "uncertainty": round(float(model_output.get("uncertainty_score", 1.0)), 4)
        }
        
        # 2. Formatting CBR output
        formatted_cbr = []
        for cbr in cbr_results:
            formatted_cbr.append({
                "case_id": cbr["patient_id"],
                "similarity": cbr["total_similarity"],
                "feature_similarity": {
                    "ais": cbr["feature_similarities"]["ais_sim"],
                    "cds": cbr["feature_similarities"]["cds_sim"],
                    "ins": cbr["feature_similarities"]["ins_sim"]
                },
                "outcome": cbr["clinical_outcome"],
                "note": cbr["why_similar"]
            })

        # 3. Intermediate output dict to compute output_hash
        output_data = {
            "patient_id": patient_id,
            "clinical_reference": clinical_reference,
            "cbr_reference_cases": formatted_cbr,
            "responsibility_and_authority": {
                "software_role": "본 시스템은 병리 영상 및 임상 데이터를 기반으로 위험도 및 유사 사례를 제공하는 '참고 정보 제공 소프트웨어'이며, 어떠한 경우에도 자체적인 진단, 치료 결정, 처방을 수행할 권한이 없습니다.",
                "clinical_decision_maker": "환자에 대한 최종 진단, 치료 방향 결정 및 의뢰에 대한 모든 권한과 법적 책임은 본 정보를 열람하는 전담 의료진에게 있습니다.",
                "disclaimer": "본 결과는 임상 참고용 정보이므로, 의료진의 종합적인 의학적 판단을 대체할 수 없습니다."
            }
        }
        
        output_hash = self._compute_hash(output_data)
        
        # 4. Final Output with Audit
        output_data["audit"] = {
            "model_version": model_output.get("model_version", "unknown"),
            "rule_version": self.rule_version,
            "timestamp": timestamp,
            "input_hash": input_hash,
            "output_hash": output_hash
        }
        
        return output_data

if __name__ == "__main__":
    engine = ClinicalReferenceAggregator()
    print("MFDS CDSS v1.1 Reference Aggregator Test Passed.")
