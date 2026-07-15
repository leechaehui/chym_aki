"""Service — 알림 생성/조회/읽음 (작업지시서 6/7.1)."""
import pytest

from core.exceptions import NotFoundError
from core.query_optimizer import Page
from services.notification_service import NotificationService


def test_create_and_list_for_department(db):
    svc = NotificationService(db)
    dept = "neph-test-dept"
    out = svc.create(
        {"department": dept, "severity": "CRITICAL", "title": "AKI 경보", "message": "고위험"}
    )
    assert out.id
    listed = svc.list_for_department(dept, Page.of(20, 0))
    assert any(n.id == out.id for n in listed)


def test_mark_read_sets_read_true(db):
    svc = NotificationService(db)
    out = svc.create(
        {"department": "d-test", "severity": "INFO", "title": "t", "message": "m"}
    )
    assert out.read is False
    updated = svc.mark_read(out.id)
    assert updated.read is True


def test_mark_read_unknown_raises_not_found(db):
    with pytest.raises(NotFoundError):
        NotificationService(db).mark_read("noti-does-not-exist")


def test_list_is_paginated(db):
    svc = NotificationService(db)
    dept = "d-page"
    for i in range(5):
        svc.create({"department": dept, "severity": "INFO", "title": f"t{i}", "message": "m"})
    page = svc.list_for_department(dept, Page.of(3, 0))
    assert len(page) <= 3
