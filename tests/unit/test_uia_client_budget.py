from src.perception.uia_client import UIAClient


class _Rect:
    left = 0
    top = 0
    right = 10
    bottom = 10


class _Control:
    Name = ""
    AutomationId = ""
    ControlTypeName = "ButtonControl"
    BoundingRectangle = _Rect()
    IsEnabled = True
    Handle = None

    def __init__(self, children=None):
        self._children = children or []

    def GetChildren(self):
        return self._children


def test_find_all_bounded_stops_at_max_elements_without_returning_zero_budget_tail(monkeypatch):
    monkeypatch.setattr("src.perception.uia_client.uia.ControlFromHandle", lambda _hwnd: _Control())
    client = UIAClient(123)
    client._root = _Control([_Control([_Control(), _Control()]), _Control()])

    result = client.find_all_bounded(max_elements=2, timeout_seconds=60.0)

    assert len(result) == 2
    assert client.find_all_bounded_diagnostics["mode"] == "bounded"
    assert client.find_all_bounded_diagnostics["truncated"] is True
    assert client.find_all_bounded_diagnostics["visited_count"] == 2
