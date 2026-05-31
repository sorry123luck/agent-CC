"""
CLI 命令行接口
提供命令行方式调用 OpenClaw Desktop Agent 功能
"""

import json
import sys
from pathlib import Path

import typer

# 确保 src 在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.indexer.catalog_service import CatalogService
from src.common.logger import setup_logger, logger

app = typer.Typer(help="OpenClaw Desktop Agent CLI")


@app.command()
def search(query: str, limit: int = 10):
    """
    搜索软件

    Args:
        query: 搜索关键词
        limit: 返回结果数量限制
    """
    setup_logger()
    logger.info(f"搜索软件: {query}")

    service = CatalogService()
    results = service.search(query)

    if not results:
        typer.echo(f"未找到匹配 '{query}' 的软件")
        return

    typer.echo(f"\n找到 {len(results)} 个匹配结果：\n")
    for i, app in enumerate(results[:limit], 1):
        typer.echo(f"{i}. {app['display_name']}")
        typer.echo(f"   规范名: {app['canonical_name']}")
        if app.get("publisher"):
            typer.echo(f"   发布者: {app['publisher']}")
        if app.get("install_location"):
            typer.echo(f"   安装位置: {app['install_location']}")
        if app.get("launch_path"):
            typer.echo(f"   启动路径: {app['launch_path']}")
        typer.echo()


@app.command()
def launch(name: str):
    """
    启动软件

    Args:
        name: 软件名称（显示名或别名）
    """
    setup_logger()
    logger.info(f"启动软件: {name}")

    service = CatalogService()
    result = service.launch(name)

    if result["success"]:
        typer.secho(f"[OK] 成功启动: {result['app_name']}", fg=typer.colors.GREEN)
        typer.echo(f"  消息: {result['message']}")
    else:
        typer.secho(f"[FAIL] 启动失败: {result['app_name']}", fg=typer.colors.RED)
        typer.echo(f"  原因: {result['message']}")


@app.command()
def scan():
    """
    执行全量软件扫描
    """
    setup_logger()
    logger.info("执行全量扫描")

    from src.storage.db import init_db
    db = init_db("data/openclaw.db")
    db.create_all()

    service = CatalogService()

    typer.echo("开始扫描...\n")
    count = service.scan_all()

    typer.secho(f"\n✓ 扫描完成！共发现 {count} 个软件", fg=typer.colors.GREEN)


@app.command()
def list_apps(limit: int = 20):
    """
    列出所有已索引的软件

    Args:
        limit: 显示数量限制
    """
    setup_logger()

    from src.storage.db import Session
    from src.storage.repositories import AppRepository

    with Session() as sess:
        repo = AppRepository(sess)
        all_apps = repo.list_all()

        if not all_apps:
            typer.echo("没有已索引的软件，请先运行 scan 命令")
            return

        typer.echo(f"\n已索引软件总数: {len(all_apps)}\n")
        typer.echo(f"显示前 {min(limit, len(all_apps))} 个：\n")

        for i, app in enumerate(all_apps[:limit], 1):
            typer.echo(f"{i}. {app.display_name}")
            typer.echo(f"   {app.canonical_name}")
            typer.echo()


@app.command()
def action_click(hwnd: int, x: int, y: int):
    """
    在窗口指定坐标点击

    Args:
        hwnd: 窗口句柄（十进制）
        x: 屏幕 X 坐标
        y: 屏幕 Y 坐标
    """
    setup_logger()
    logger.info(f"执行点击: hwnd={hwnd}, ({x}, {y})")

    from src.execution.action_service import ActionService

    svc = ActionService()
    result = svc.action_click(hwnd, x, y)

    if result.success:
        typer.secho(f"[OK] 点击成功 ({x}, {y})", fg=typer.colors.GREEN)
        typer.echo(f"  策略: {result.strategy}, 尝试次数: {result.attempts}")
    else:
        typer.secho(f"[FAIL] 点击失败", fg=typer.colors.RED)
        typer.echo(f"  错误: {result.error}")


@app.command()
def list_windows():
    """
    列出当前所有窗口
    """
    setup_logger()

    from src.windows.window_enum import WindowEnumService

    svc = WindowEnumService()
    windows = svc.enumerate_all(refresh=True)

    typer.echo(f"\n当前窗口总数: {len(windows)}\n")
    for i, w in enumerate(windows, 1):
        title = w.title or "(无标题)"
        if len(title) > 60:
            title = title[:60] + "..."
        typer.echo(f"{i}. {title}")
        typer.echo(f"   hwnd={w.hwnd}, rect={w.rect}")
        typer.echo()


@app.command()
def inspect_window(
    hwnd: int,
    output_dir: str = "data/inspect",
    include_ocr: bool = True,
):
    """
    导出指定窗口的最小 Inspector 检查包。
    """
    setup_logger()

    from src.inspector.service import InspectorService

    svc = InspectorService()
    result = svc.inspect_window(
        hwnd=hwnd,
        output_dir=output_dir,
        include_ocr=include_ocr,
    )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def inspect_foreground(
    output_dir: str = "data/inspect",
    include_ocr: bool = True,
):
    """
    导出当前前台窗口的最小 Inspector 检查包。
    """
    setup_logger()

    from src.inspector.service import InspectorService

    svc = InspectorService()
    result = svc.inspect_foreground_window(
        output_dir=output_dir,
        include_ocr=include_ocr,
    )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def build_openclaw_payload(
    snapshot_path: str,
    task: str,
    output_path: str | None = None,
    max_candidates: int = 120,
):
    """
    根据 snapshot.json 构建给 OpenClaw 的弱结构输入。
    """
    setup_logger()

    from src.inspector.service import InspectorService

    svc = InspectorService()
    result = svc.build_openclaw_payload(
        snapshot_path=snapshot_path,
        task=task,
        max_candidates=max_candidates,
    )
    payload = result["openclaw_payload"]
    if output_path:
        Path(output_path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def export_candidate_regression(
    snapshot_path: str,
    task: str,
    image_path: str | None = None,
    output_dir: str | None = None,
    max_candidates: int = 120,
    draw_labels: bool = False,
):
    """
    基于固定 snapshot/image 导出本地候选回归包。
    """
    setup_logger()

    from src.inspector.service import InspectorService

    svc = InspectorService()
    result = svc.export_candidate_regression(
        snapshot_path=snapshot_path,
        task=task,
        image_path=image_path,
        output_dir=output_dir,
        max_candidates=max_candidates,
        draw_labels=draw_labels,
    )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def export_real_app_regressions(
    baseline_dir: str = "data/baselines/real_apps",
    task: str = "分析当前页面中的可交互候选",
    max_candidates: int = 160,
    draw_labels: bool = False,
):
    """
    批量导出真实软件样本的候选回归结果与摘要。
    """
    setup_logger()

    from src.inspector.service import InspectorService

    svc = InspectorService()
    result = svc.export_real_app_regressions(
        baseline_dir=baseline_dir,
        task=task,
        max_candidates=max_candidates,
        draw_labels=draw_labels,
    )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def normalize_decision_record(
    decision_path: str,
    output_path: str | None = None,
):
    """
    将 OpenClaw 返回的 JSON 标准化为仓库内 decision_record 结构。
    """
    setup_logger()

    from src.inspector.service import InspectorService

    svc = InspectorService()
    result = svc.normalize_decision_record(decision_path=decision_path)
    record = result["decision_record"]
    if output_path:
        Path(output_path).write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def build_recrop_request(
    snapshot_path: str,
    decision_path: str,
    output_path: str | None = None,
    crop_scale: float = 1.5,
):
    """
    根据 snapshot 与 decision_record 生成局部重采样请求。
    """
    setup_logger()

    from src.inspector.service import InspectorService

    svc = InspectorService()
    result = svc.build_recrop_request(
        snapshot_path=snapshot_path,
        decision_path=decision_path,
        crop_scale=crop_scale,
    )
    recrop_request = result.get("recrop_request")
    if output_path and recrop_request is not None:
        Path(output_path).write_text(
            json.dumps(recrop_request, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def reanalyze_focus_region(
    snapshot_path: str,
    decision_path: str,
    task: str,
    output_path: str | None = None,
    image_path: str | None = None,
    crop_scale: float = 1.5,
    max_candidates: int = 120,
):
    """
    根据 focus_bbox 对已有截图局部裁图，重跑 OCR / vision，并生成第二轮 OpenClaw payload。
    """
    setup_logger()

    from src.inspector.service import InspectorService

    svc = InspectorService()
    result = svc.reanalyze_focus_region(
        snapshot_path=snapshot_path,
        decision_path=decision_path,
        task=task,
        image_path=image_path,
        crop_scale=crop_scale,
        max_candidates=max_candidates,
    )
    payload = result.get("focused_openclaw_payload")
    if output_path and payload is not None:
        Path(output_path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
