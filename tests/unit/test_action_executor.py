"""
ActionExecutor 单元测试
"""
import pytest
from unittest.mock import MagicMock, patch
import win32con
from src.execution.action_executor import (
    ActionExecutor, ActionResult, MouseButton, POINT
)


class TestActionExecutor:
    """ActionExecutor 测试"""

    def test_click_success(self):
        """点击成功"""
        executor = ActionExecutor()

        with patch.object(executor._activator, "activate", return_value=True):
            with patch("win32api.SetCursorPos"):
                with patch("win32api.mouse_event"):
                    result = executor.click(hwnd=12345, x=100, y=200)

        assert result.success is True
        assert result.action == "click"
        assert "100" in result.details
        assert "200" in result.details

    def test_click_activate_failure(self):
        """窗口激活失败"""
        executor = ActionExecutor()

        with patch.object(executor._activator, "activate", return_value=False):
            result = executor.click(hwnd=12345, x=100, y=200)

        assert result.success is False
        assert result.action == "click"

    def test_click_double(self):
        """双击"""
        executor = ActionExecutor()

        with patch.object(executor._activator, "activate", return_value=True):
            with patch("win32api.SetCursorPos"):
                with patch("win32api.mouse_event"):
                    result = executor.click(hwnd=12345, x=100, y=200, double=True)

        assert result.success is True

    def test_action_result_dataclass(self):
        """ActionResult 数据类"""
        result = ActionResult(success=True, action="test", details="ok")
        assert result.success is True
        assert result.action == "test"
        assert result.details == "ok"
        assert result.error is None

    def test_mouse_button_enum(self):
        """鼠标按钮枚举"""
        assert MouseButton.LEFT.value == "left"
        assert MouseButton.RIGHT.value == "right"
        assert MouseButton.MIDDLE.value == "middle"

    def test_point_structure(self):
        """POINT 结构"""
        pt = POINT(100, 200)
        assert pt.x == 100
        assert pt.y == 200

    def test_send_keys_supports_modifier_with_special_key(self):
        """send_keys 支持 Ctrl+Enter 这类修饰键 + 特殊键组合"""
        executor = ActionExecutor()

        with patch.object(executor._activator, "activate", return_value=True), patch(
            "win32api.keybd_event"
        ) as mock_keybd_event:
            result = executor.send_keys(hwnd=12345, keys="^{ENTER}")

        assert result.success is True
        key_events = [call.args[0] for call in mock_keybd_event.call_args_list]
        assert win32con.VK_CONTROL in key_events
        assert win32con.VK_RETURN in key_events


class TestActionVerifier:
    """ActionVerifier 测试"""

    def test_verify_window_state_timeout(self):
        """窗口状态校验超时"""
        from src.execution.action_verifier import ActionVerifier

        verifier = ActionVerifier()

        with patch("win32gui.IsWindow", return_value=True):
            with patch("win32gui.IsIconic", return_value=False):
                with patch("win32gui.GetWindowPlacement", return_value=(0, 1, 0, 0, 0)):
                    with patch.object(verifier._window_enum, "get_foreground_window", return_value=None):
                        result = verifier.verify_window_state(
                            hwnd=12345,
                            expected_state="active",
                            timeout=0.2,
                        )

        assert result.verified is False
        assert result.method.value == "window_state"


class TestFailureRecovery:
    """FailureRecovery 测试"""

    def test_retry_config(self):
        """重试配置"""
        from src.execution.failure_recovery import RetryConfig

        config = RetryConfig(max_attempts=3, initial_delay=0.1)
        assert config.max_attempts == 3
        assert config.initial_delay == 0.1

    def test_fallback_config(self):
        """备用方案配置"""
        from src.execution.failure_recovery import FallbackConfig

        config = FallbackConfig(
            fallback_coordinates=[(100, 200), (150, 250)],
            fallback_automation_ids=["btn_ok", "btn_cancel"],
        )
        assert len(config.fallback_coordinates) == 2
        assert len(config.fallback_automation_ids) == 2

    def test_recovery_result(self):
        """恢复结果"""
        from src.execution.failure_recovery import RecoveryResult, RecoveryStrategy

        result = RecoveryResult(
            recovered=True,
            strategy_used=RecoveryStrategy.RETRY,
            attempts=2,
        )
        assert result.recovered is True
        assert result.strategy_used == RecoveryStrategy.RETRY
        assert result.attempts == 2

    def test_fallback_coordinates_used_after_retry(self):
        """备用坐标在重试失败后被使用"""
        from src.execution.failure_recovery import FailureRecovery, FallbackConfig, RecoveryStrategy, RetryConfig

        recovery = FailureRecovery(retry_config=RetryConfig(max_attempts=1, initial_delay=0.01))
        click_results = []

        def mock_click(hwnd, x, y):
            click_results.append((x, y))
            # First call fails, second call with fallback coords succeeds
            if len(click_results) == 1:
                return MagicMock(success=False, error="fail")
            return MagicMock(success=True, details="ok")

        with patch.object(recovery._executor, "click", side_effect=mock_click):
            result = recovery.click_with_recovery(
                hwnd=12345,
                x=100,
                y=200,
                fallback_config=FallbackConfig(
                    fallback_coordinates=[(150, 250), (300, 350)]
                ),
            )

        assert result.recovered is True
        assert result.strategy_used == RecoveryStrategy.FALLBACK_COORDINATE
        # Verify fallback coordinates were actually used (not original)
        assert click_results == [(100, 200), (150, 250)]

    def test_fallback_element_after_coordinates_fail(self):
        """备用元素在前两种策略失败后被使用"""
        from src.execution.failure_recovery import FailureRecovery, FallbackConfig, RecoveryStrategy, RetryConfig

        recovery = FailureRecovery(retry_config=RetryConfig(max_attempts=1, initial_delay=0.01))
        calls = []

        def mock_click(hwnd, x, y):
            calls.append(("click", x, y))
            return MagicMock(success=False, error="fail")

        def mock_invoke(hwnd, automation_id=None, **kwargs):
            calls.append(("invoke", automation_id))
            return MagicMock(success=True, details="ok")

        with patch.object(recovery._executor, "click", side_effect=mock_click):
            with patch.object(recovery._executor, "invoke_uia_element", side_effect=mock_invoke):
                result = recovery.click_with_recovery(
                    hwnd=12345,
                    x=100,
                    y=200,
                    fallback_config=FallbackConfig(
                        fallback_coordinates=[(150, 250)],
                        fallback_automation_ids=["btn_ok"],
                    ),
                )

        assert result.recovered is True
        assert result.strategy_used == RecoveryStrategy.FALLBACK_ELEMENT
        # Verify: retry (100,200) -> fallback_coord (150,250) -> fallback_elem (btn_ok)
        assert calls == [
            ("click", 100, 200),
            ("click", 150, 250),
            ("invoke", "btn_ok"),
        ]

    def test_all_strategies_fail_aborts(self):
        """所有策略都失败时中止"""
        from src.execution.failure_recovery import FailureRecovery, FallbackConfig, RecoveryStrategy, RetryConfig

        recovery = FailureRecovery(retry_config=RetryConfig(max_attempts=1, initial_delay=0.01))

        def mock_click(hwnd, x, y):
            return MagicMock(success=False, error="fail")

        with patch.object(recovery._executor, "click", side_effect=mock_click):
            with patch.object(recovery._executor, "invoke_uia_element", return_value=MagicMock(success=False, error="fail")):
                result = recovery.click_with_recovery(
                    hwnd=12345,
                    x=100,
                    y=200,
                    fallback_config=FallbackConfig(
                        fallback_coordinates=[(150, 250)],
                        fallback_automation_ids=["btn_ok"],
                    ),
                )

        assert result.recovered is False
        assert result.strategy_used == RecoveryStrategy.ABORT


class TestActionService:
    """ActionService 测试"""

    def test_action_outcome(self):
        """ActionOutcome 数据类"""
        from src.execution.action_service import ActionOutcome

        outcome = ActionOutcome(
            success=True,
            action="click",
            message="ok",
            attempts=1,
            strategy="direct",
        )
        assert outcome.success is True
        assert outcome.action == "click"
        assert outcome.attempts == 1
        assert outcome.strategy == "direct"

    def test_click_without_recovery_success(self):
        """点击成功（无恢复，直接执行）"""
        from src.execution.action_service import ActionService

        svc = ActionService()

        with patch.object(svc._executor, "click") as mock_click:
            mock_click.return_value = MagicMock(success=True, details="ok", error=None)
            outcome = svc.action_click(hwnd=12345, x=100, y=200, verify=False, with_recovery=False)

        assert outcome.success is True
        assert outcome.strategy == "direct"
        assert outcome.attempts == 1

    def test_click_with_recovery_verification_failure_triggers_retry(self):
        """校验失败触发重试"""
        from src.execution.action_service import ActionService
        from src.execution.failure_recovery import RecoveryResult, RecoveryStrategy

        svc = ActionService()

        call_count = [0]

        def mock_click(hwnd, x, y, button=None, double=False):
            call_count[0] += 1
            return MagicMock(success=True, details=f"click {call_count[0]}", error=None)

        def mock_verify(hwnd, state, timeout=None):
            if call_count[0] < 2:
                return MagicMock(verified=False)
            return MagicMock(verified=True)

        with patch.object(svc._executor, "click", side_effect=mock_click):
            with patch.object(svc._verifier, "verify_window_state", side_effect=mock_verify):
                with patch.object(svc._recovery._executor, "click", side_effect=mock_click):
                    with patch.object(svc._recovery._verifier, "verify_window_state", side_effect=mock_verify):
                        outcome = svc.action_click(hwnd=12345, x=100, y=200, verify=True)

        # Should have retried until verification passed
        assert outcome.attempts >= 1
