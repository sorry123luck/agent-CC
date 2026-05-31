"""Tests for filtering root UIA shells from virtual model output."""

from src.integration.api_server import _is_root_shell_virtual_candidate


def test_full_window_title_bar_shell_is_filtered():
    assert _is_root_shell_virtual_candidate(
        canonical_role="title_bar",
        relative_bounds=[0.0, 0.0, 1.0, 1.0],
        provider_sources=["uia"],
    ) is True


def test_full_window_sidebar_shell_is_filtered():
    assert _is_root_shell_virtual_candidate(
        canonical_role="sidebar",
        relative_bounds=[0.007, 0.0, 0.992, 0.989],
        provider_sources=["uia"],
    ) is True


def test_real_sidebar_is_not_filtered():
    assert _is_root_shell_virtual_candidate(
        canonical_role="sidebar",
        relative_bounds=[0.0, 0.08, 0.28, 0.94],
        provider_sources=["uia"],
    ) is False


def test_manual_full_window_region_is_not_filtered():
    assert _is_root_shell_virtual_candidate(
        canonical_role="layout",
        relative_bounds=[0.0, 0.0, 1.0, 1.0],
        provider_sources=["manual"],
    ) is False
