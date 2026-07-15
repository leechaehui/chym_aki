from services import alert_service


def test_build_candidate_skips_when_subject_id_missing():
    event = {"stage": "CONFIRMED", "patientId": "patient-1", "subjectId": None}

    assert alert_service._build_candidate(event) is None


def test_build_candidate_accepts_when_subject_id_present():
    event = {"stage": "SUSPECTED", "patientId": "patient-2", "subject_id": 123456}

    candidate = alert_service._build_candidate(event)

    assert candidate is not None
    assert candidate["subject_id"] == 123456
