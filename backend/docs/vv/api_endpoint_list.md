# API Endpoint List (작업지시서 9)

총 **35** 개 엔드포인트. OpenAPI(`app.openapi()`)에서 자동 추출.

| Tag | Method | Path | Summary |
|---|---|---|---|
| audit | GET | `/api/audit` | List Audit Logs |
| auth | GET | `/api/auth/accounts` | List Accounts |
| auth | PATCH | `/api/auth/accounts/{user_id}/approval` | Set Approval |
| auth | POST | `/api/auth/login` | Login |
| auth | GET | `/api/auth/me` | Me |
| auth | POST | `/api/auth/signup` | Sign Up |
| beds | GET | `/api/beds` | List Beds |
| beds | GET | `/api/beds/details` | List Bed Details |
| beds | GET | `/api/beds/emergency-patients` | List Emergency Patients |
| beds | GET | `/api/beds/reservations` | List Bed Reservations |
| beds | GET | `/api/beds/summary` | Bed Summary |
| beds | POST | `/api/beds/{bed_id}/assign` | Assign Bed |
| beds | POST | `/api/beds/{bed_id}/release` | Release Bed |
| consultation | GET | `/api/consultations` | List Consults |
| consultation | POST | `/api/consultations` | Request Consult |
| consultation | GET | `/api/consultations/{consult_id}` | Get Consult |
| consultation | POST | `/api/consultations/{consult_id}/accept` | Accept Consult |
| consultation | POST | `/api/consultations/{consult_id}/reply` | Reply Consult |
| nephrology | POST | `/api/nephrology/aki/analyze` | Analyze Features |
| nephrology | POST | `/api/nephrology/aki/analyze/{patient_id}` | Analyze Patient |
| nephrology | POST | `/api/nephrology/aki/predict-vector` | Predict Vector |
| nephrology | GET | `/api/nephrology/timeline/{patient_id}` | Read Timeline |
| notification | GET | `/api/notifications` | List Notifications |
| notification | POST | `/api/notifications` | Create Notification |
| notification | POST | `/api/notifications/{notification_id}/read` | Mark Read |
| pathology | GET | `/api/pathology` | List Results |
| pathology | GET | `/api/pathology/by-consult/{consult_id}` | Get By Consult |
| patients | GET | `/api/patients` | List Patients |
| patients | GET | `/api/patients/{patient_id}` | Get Patient |
| system | GET | `/health` | Health |
| timeline | POST | `/api/timeline/event` | Create Event |
| timeline | GET | `/api/timeline/patient/{patient_id}` | Get Patient Timeline |
| voice | POST | `/api/voice/draft` | Generate Draft |
| voice | GET | `/api/voice/drafts/{patient_id}` | List Drafts |
| voice | POST | `/api/voice/transcribe` | Transcribe |
