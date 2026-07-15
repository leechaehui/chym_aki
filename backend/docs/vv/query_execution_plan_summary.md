# Query Execution Plan Summary (Verification 3.2)

**PostgreSQL** `EXPLAIN (ANALYZE, BUFFERS)` 기준(실제 실행 후 계획).
- **인덱스 사용 가능**: `enable_seqscan=off` 강제 시 인덱스 노드 사용 여부(인덱스 존재·유효성 검증).
- **실제 plan**: 현재 시드(소규모)에서 planner 의 자연 선택. 소규모 테이블은 Seq Scan 이 더 저렴해 정상.

| 쿼리 | 인덱스 사용 가능 | 실제 plan(자연 선택) 발췌 |
|---|---|---|
| 환자 목록 (admitted_at DESC, 페이지네이션) | PASS (index 사용 가능) | Limit  (cost=1.11..1.12 rows=5 width=127) (actual time=0.019..0.020 rows=5 loops=1) |
| 환자 타임라인 (patient_id + severity, event_time DESC) | PASS (index 사용 가능) | Limit  (cost=0.01..0.02 rows=1 width=926) (actual time=0.003..0.004 rows=0 loops=1) |
| 병상 보드 (zone, label 정렬) | PASS (index 사용 가능) | Sort  (cost=1.87..1.94 rows=26 width=132) (actual time=0.034..0.035 rows=26 loops=1) |
| 협진 목록 (kind + status) | PASS (index 사용 가능) | Seq Scan on consultations  (cost=0.00..0.00 rows=1 width=1710) (actual time=0.002..0.002 rows=0 loops=1) |
| 환자 단건 (PK) | PASS (index 사용 가능) | Seq Scan on patients  (cost=0.00..1.06 rows=1 width=127) (actual time=0.007..0.007 rows=0 loops=1) |

## N+1 회피 검증 (상세 조회)

`PatientRepository.get_with_details` 는 `selectinload(labs/trend/urine)` 로,
행 수와 무관하게 **부모 1 + 자식 3 = 상수 4개 쿼리**만 실행한다(자식 IN 절 일괄 적재).
→ 환자 N명을 순회해도 4N 이 아니라 4 (단건) / 1+3 IN (목록) 으로 N+1 이 발생하지 않는다.

## 원칙 점검

- SELECT * 회피: 목록은 `load_only`/요약 DTO, 상세는 명시적 selectinload.
- 페이지네이션: 모든 목록은 `paginate()`(LIMIT/OFFSET, MAX 100) 경유.
- raw SQL: 애플리케이션 코드에 raw SQL 없음(repository layer 가 ORM 으로 캡슐화).
- aggregation: DB 레벨 처리(`bed_service.summary` 등 count 집계는 쿼리에서 수행).
- 동시성: 병상 배정은 `SELECT ... FOR UPDATE`(PostgreSQL 실제 행잠금)로 경쟁 차단.

_종합: 모든 지배 쿼리에 사용 가능한 인덱스 존재(enable_seqscan=off 검증)_
