from threading import Lock

_active: dict = {}
_lock = Lock()


def add_active_request(request_id: str, info: dict) -> None:
    with _lock:
        _active[request_id] = info


def remove_active_request(request_id: str) -> None:
    with _lock:
        _active.pop(request_id, None)


def get_active_requests() -> dict:
    with _lock:
        return dict(_active)
