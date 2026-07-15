"""환자 + 검사 데이터 엔티티 (신장내과 도메인).

책임: 환자 마스터와 그에 종속된 검사값/추이/소변량을 정규화 테이블로 보관.
- labs/trend/urine 은 1:N 자식 → selectinload 로 N+1 없이 적재(query_optimizer 참고).
- mimic_subject_id 로 AKI 코호트(MIMIC) 연계가 가능하다(선택).
"""
from sqlalchemy import Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    mrn: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(60), nullable=False)
    sex: Mapped[str] = mapped_column(String(1), nullable=False)  # M | F
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    diagnosis: Mapped[str] = mapped_column(String(200), nullable=False)
    admitted_at: Mapped[str] = mapped_column(String(40), nullable=False)
    attending: Mapped[str] = mapped_column(String(60), nullable=False)
    room: Mapped[str] = mapped_column(String(80), nullable=False)
    # AI 종합 위험 점수(0–100). AKI 모델 출력 또는 rule-based 폴백으로 갱신.
    ai_risk_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    mimic_subject_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # 환자 목록의 지배적 정렬(admitted_at DESC) 인덱스 — full scan + temp B-tree 제거.
    # (verify_queries.py EXPLAIN 검증에서 도출)
    __table_args__ = (Index("ix_patients_admitted_at", "admitted_at"),)

    labs: Mapped[list["PatientLab"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan", order_by="PatientLab.seq"
    )
    trend: Mapped[list["PatientTrendPoint"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="PatientTrendPoint.date",
    )
    urine_output: Mapped[list["PatientUrinePoint"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="PatientUrinePoint.date",
    )


class PatientLab(Base):
    """환자별 검사 항목 한 건(정상범위/플래그 포함)."""

    __tablename__ = "patient_labs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    key: Mapped[str] = mapped_column(String(20), nullable=False)
    label: Mapped[str] = mapped_column(String(40), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)
    ref_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    ref_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    flag: Mapped[str] = mapped_column(String(10), nullable=False)  # normal|high|low

    patient: Mapped["Patient"] = relationship(back_populates="labs")

    __table_args__ = (Index("ix_patient_labs_patient", "patient_id"),)


class PatientTrendPoint(Base):
    """일자별 주요 신기능 추이(Cr/eGFR/BUN)."""

    __tablename__ = "patient_trend_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD
    creatinine: Mapped[float] = mapped_column(Float, nullable=False)
    egfr: Mapped[float] = mapped_column(Float, nullable=False)
    bun: Mapped[float] = mapped_column(Float, nullable=False)

    patient: Mapped["Patient"] = relationship(back_populates="trend")

    __table_args__ = (Index("ix_patient_trend_patient", "patient_id"),)


class PatientUrinePoint(Base):
    """일자별 시간당 소변량(mL/kg/hr) — 핍뇨/무뇨 판정 근거."""

    __tablename__ = "patient_urine_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[str] = mapped_column(String(10), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    patient: Mapped["Patient"] = relationship(back_populates="urine_output")

    __table_args__ = (Index("ix_patient_urine_patient", "patient_id"),)
