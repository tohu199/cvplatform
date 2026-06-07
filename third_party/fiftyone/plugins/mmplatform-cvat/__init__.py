"""mmplatform FiftyOne plugin: k-center ordering + CVAT upload."""

from __future__ import annotations

import sys
import time

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

from .kcenter import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SELECT,
    EMBEDDING_MODEL_OPTIONS,
    KCenterProgress,
    MIN_SELECT,
    RANK_FIELD,
    compute_kcenter_order,
    normalize_embedding_model_name,
    selection_bounds_message,
    set_kcenter_ranks,
    validate_selection_count,
)

PANEL_NAME = "cvat_kcenter_panel"
PANEL_LABEL = "CVAT: k-center 選定"


def _panel_config() -> foo.PanelConfig:
    cfg = foo.PanelConfig(
        name=PANEL_NAME,
        label=PANEL_LABEL,
        icon="cloud_upload",
    )
    cfg.unlisted = False
    return cfg


def _parse_classes(raw: str) -> list[str]:
    classes = [part.strip() for part in (raw or "").split(",") if part.strip()]
    if not classes:
        raise ValueError("クラス名を 1 つ以上指定してください（カンマ区切り）")
    return classes


def _default_anno_key() -> str:
    return f"mmplatform_cvat_{int(time.time())}"


def _embedding_model_dropdown() -> types.DropdownView:
    dropdown = types.DropdownView()
    for option in EMBEDDING_MODEL_OPTIONS:
        dropdown.add_choice(
            option["name"],
            label=option["label"],
            description=option["description"],
        )
    return dropdown


def _python_version_note() -> str:
    py = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info < (3, 9):
        return (
            f"現在 Python {py} のため DINOv2 は使えません。"
            "DINO v1 / MobileNet / ResNet を選んでください。"
        )
    return f"Python {py}"


class CvatKCenterPanel(foo.Panel):
    @property
    def config(self):
        return _panel_config()

    def on_load(self, ctx):
        ctx.panel.state.anno_key = ctx.panel.get_state("anno_key", _default_anno_key())
        ctx.panel.state.label_field = ctx.panel.get_state("label_field", "ground_truth")
        ctx.panel.state.classes = ctx.panel.get_state("classes", "person,car,dog")
        ctx.panel.state.embedding_model = ctx.panel.get_state(
            "embedding_model", DEFAULT_EMBEDDING_MODEL
        )
        ctx.panel.state.status = ctx.panel.get_state(
            "status",
            "k-center 順に並べ替えてから CVAT に送れます。",
        )
        ctx.panel.state.kcenter_running = ctx.panel.get_state("kcenter_running", False)
        ctx.panel.state.kcenter_progress = ctx.panel.get_state("kcenter_progress", 0.0)
        ctx.panel.state.kcenter_progress_label = ctx.panel.get_state(
            "kcenter_progress_label", ""
        )
        ctx.panel.state.kcenter_log = ctx.panel.get_state("kcenter_log", "")
        ctx.panel.state.show_kcenter_log = ctx.panel.get_state("show_kcenter_log", False)

    def _kcenter_progress_hooks(self, ctx) -> KCenterProgress:
        def on_progress(fraction: float, label: str) -> None:
            ctx.panel.state.kcenter_running = True
            ctx.panel.state.kcenter_progress = fraction
            ctx.panel.state.kcenter_progress_label = label
            ctx.panel.state.status = label
            ctx.ops.set_progress(label=label, progress=fraction)

        def on_log(text: str) -> None:
            ctx.panel.state.kcenter_log = text

        return KCenterProgress(on_progress=on_progress, on_log=on_log)

    def toggle_kcenter_log(self, ctx):
        ctx.panel.state.show_kcenter_log = not bool(
            ctx.panel.get_state("show_kcenter_log", False)
        )

    def on_embedding_model_change(self, ctx):
        value = ctx.params.get("value")
        if value:
            ctx.panel.state.embedding_model = value

    def _target_view(self, ctx):
        if ctx.view != ctx.dataset.view():
            return ctx.view
        return ctx.dataset

    def order_and_select(self, ctx):
        target = self._target_view(ctx)
        count = target.count()
        if count == 0:
            ctx.ops.notify("サンプルがありません", variant="error")
            return

        try:
            model_name = normalize_embedding_model_name(ctx.panel.state.embedding_model)
        except ValueError as exc:
            ctx.ops.notify(str(exc), variant="error")
            return

        ctx.panel.state.status = f"embedding 計算中…（{model_name}）"
        ctx.panel.state.kcenter_running = True
        ctx.panel.state.kcenter_progress = 0.0
        ctx.panel.state.kcenter_progress_label = "k-center 処理を開始…"
        ctx.panel.state.kcenter_log = ""
        ctx.panel.state.show_kcenter_log = False

        progress = self._kcenter_progress_hooks(ctx)
        try:
            progress.log(f"モデル: {model_name}")
            progress.log(f"対象サンプル数: {count}")

            result = compute_kcenter_order(
                ctx.dataset, target, model_name, progress=progress
            )
            progress.set(0.98, "ビューを更新中…")
            set_kcenter_ranks(ctx.dataset, result.ordered_ids)
            sorted_view = target.sort_by(RANK_FIELD)
            ctx.ops.set_view(sorted_view)

            default_n = min(DEFAULT_SELECT, len(result.ordered_ids))
            ctx.ops.set_selected_samples(result.ordered_ids[:default_n])

            summary = (
                f"{len(result.ordered_ids)} 枚を k-center 順に並べ替え、"
                f"先頭 {default_n} 枚を選択しました。"
                f"（embedding: {result.model_name}）"
            )
            progress.set(1.0, "完了", force=True)
            progress.log(summary)
            ctx.panel.state.kcenter_running = False
            ctx.panel.state.status = summary
            ctx.ops.set_progress(label="k-center 完了", progress=1.0)
            ctx.ops.notify(summary, variant="success")
        except Exception as exc:
            progress.log(f"ERROR: {exc}")
            ctx.panel.state.kcenter_running = False
            ctx.panel.state.status = f"失敗: {exc}"
            ctx.ops.set_progress(label=f"失敗: {exc}", progress=0.0)
            ctx.ops.notify(str(exc), variant="error")

    def select_default(self, ctx):
        target = self._target_view(ctx)
        ids = [sample.id for sample in target.sort_by(RANK_FIELD).limit(DEFAULT_SELECT)]
        if not ids:
            ctx.ops.notify(
                "k-center 順が未計算です。先に「k-center で並べ替え」を実行してください。",
                variant="warning",
            )
            return

        ctx.ops.set_selected_samples(ids)
        ctx.panel.state.status = f"先頭 {len(ids)} 枚を選択しました。"
        ctx.ops.notify(ctx.panel.state.status, variant="info")

    def send_to_cvat(self, ctx):
        selected = list(ctx.selected or [])
        ok, message = validate_selection_count(len(selected))
        if not ok:
            ctx.ops.notify(message, variant="error")
            ctx.panel.state.status = message
            return

        anno_key = (ctx.panel.state.anno_key or _default_anno_key()).strip()
        label_field = (ctx.panel.state.label_field or "ground_truth").strip()
        classes = _parse_classes(ctx.panel.state.classes)

        view = ctx.dataset.select(selected)
        ctx.panel.state.status = f"CVAT へ {len(selected)} 枚を送信中…"
        try:
            view.annotate(
                anno_key,
                backend="cvat",
                label_field=label_field,
                label_type="detections",
                classes=classes,
                launch_editor=True,
            )
            ctx.panel.state.status = (
                f"CVAT 送信完了: anno_key='{anno_key}', {len(selected)} 枚"
            )
            ctx.ops.notify(ctx.panel.state.status, variant="success")
        except Exception as exc:
            ctx.panel.state.status = f"CVAT 送信失敗: {exc}"
            ctx.ops.notify(str(exc), variant="error")

    def render(self, ctx):
        panel = types.Object()

        panel.md(
            f"""
### CVAT k-center 選定

1. **embedding モデルを選択**
2. **k-center で並べ替え** — 多様性順に並べ、先頭 {DEFAULT_SELECT} 枚を選択
3. グリッドで {MIN_SELECT} 枚以上を選択
4. **CVAT に送信**

{_python_version_note()}

{selection_bounds_message(len(ctx.selected or []))}
            """.strip(),
            name="intro",
        )

        panel.str(
            "embedding_model",
            label="Embedding model",
            view=_embedding_model_dropdown(),
            on_change=self.on_embedding_model_change,
            required=True,
        )
        panel.str("anno_key", label="Annotation key", required=True)
        panel.str("label_field", label="Label field", required=True)
        panel.str(
            "classes",
            label="Classes (comma-separated)",
            description="例: person,car,dog",
            required=True,
        )
        panel.message("status", ctx.panel.state.status or "")

        running = bool(ctx.panel.state.kcenter_running)
        progress_value = float(ctx.panel.state.kcenter_progress or 0.0)
        if running or progress_value > 0:
            progress_label = (
                ctx.panel.state.kcenter_progress_label
                or ("処理中…" if running else "前回の処理")
            )
            progress_view = types.ProgressView(label=progress_label)
            panel.float("kcenter_progress", view=progress_view)

        log_text = ctx.panel.state.kcenter_log or ""
        if log_text:
            show_log = bool(ctx.panel.state.show_kcenter_log)
            log_lines = len(log_text.splitlines())
            panel.btn(
                "toggle_kcenter_log_btn",
                label="詳細ログを隠す" if show_log else f"詳細ログを表示 ({log_lines} 行)",
                icon="article" if not show_log else "expand_less",
                on_click=self.toggle_kcenter_log,
                variant="outlined",
            )
            if show_log:
                panel.str(
                    "kcenter_log",
                    label="詳細ログ",
                    view=types.CodeView(language="text"),
                )

        panel.btn(
            "order_btn",
            label="k-center で並べ替え",
            icon="sort",
            on_click=self.order_and_select,
            variant="contained",
        )
        panel.btn(
            "select_default_btn",
            label=f"先頭 {DEFAULT_SELECT} 枚を選択",
            icon="check_box",
            on_click=self.select_default,
        )
        panel.btn(
            "send_btn",
            label="CVAT に送信",
            icon="cloud_upload",
            on_click=self.send_to_cvat,
            variant="contained",
            color="primary",
        )

        return types.Property(
            panel,
            view=types.GridView(
                align_x="stretch",
                orientation="vertical",
                gap=2,
                padding=1,
            ),
        )


class SendSelectedToCvat(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="send_selected_to_cvat",
            label="Send selected to CVAT",
            dynamic=True,
            allow_delegated_execution=True,
            allow_immediate_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        selected_count = len(ctx.selected or [])
        ok, message = validate_selection_count(selected_count)
        inputs.view("bounds", types.Notice(label=message))
        if not ok and selected_count > 0:
            inputs.view(
                "count_error",
                types.Error(
                    label="Selection count",
                    description=message,
                ),
            )

        inputs.str(
            "anno_key",
            label="Annotation key",
            required=True,
            default=_default_anno_key(),
        )
        inputs.str(
            "label_field",
            label="Label field",
            required=True,
            default="ground_truth",
        )
        inputs.str(
            "classes",
            label="Classes (comma-separated)",
            required=True,
            default="person,car,dog",
        )

        header = "Send selected samples to CVAT"
        return types.Property(inputs, view=types.View(label=header))

    def execute(self, ctx):
        selected = list(ctx.selected or [])
        ok, message = validate_selection_count(len(selected))
        if not ok:
            raise ValueError(message)

        anno_key = ctx.params["anno_key"].strip()
        label_field = ctx.params["label_field"].strip()
        classes = _parse_classes(ctx.params["classes"])

        view = ctx.dataset.select(selected)
        view.annotate(
            anno_key,
            backend="cvat",
            label_field=label_field,
            label_type="detections",
            classes=classes,
            launch_editor=True,
        )

        return {"message": f"Sent {len(selected)} samples to CVAT (anno_key={anno_key})"}


class OpenCvatKcenterPanel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="open_cvat_kcenter_panel",
            label="Open CVAT k-center panel",
            description="Open the CVAT k-center selection panel in the App",
            icon="cloud_upload",
            dynamic=False,
        )

    def execute(self, ctx):
        ctx.ops.open_panel(PANEL_NAME, layout="horizontal")
        ctx.ops.notify(f"Opened panel: {PANEL_LABEL}", variant="info")
        return {"message": f"Opened {PANEL_NAME}"}


def register(p):
    p.register(CvatKCenterPanel)
    p.register(SendSelectedToCvat)
    p.register(OpenCvatKcenterPanel)
