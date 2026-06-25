"""mmplatform FiftyOne plugin: k-center / PPAL ordering + CVAT upload."""

from __future__ import annotations

import os
import sys
from typing import Optional

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
from .cvat_send import (
    CVAT_SENT_TAG,
    cvat_env_summary,
    cvat_sent_status_message,
    default_anno_key,
    send_samples_to_cvat,
    sync_cvat_sent_tags,
)
from .ppal_select import (
    PPAL_RANK_FIELD,
    classes_csv,
    compute_ppal_order,
    list_ppal_checkpoint_options,
    normalize_ppal_checkpoint,
    read_checkpoint_classes,
    set_ppal_ranks,
)

PANEL_NAME = "cvat_kcenter_panel"
PANEL_LABEL = "CVAT: 能動学習選定"

MODE_KCENTER = "kcenter"
MODE_PPAL = "ppal"
SELECTION_MODES = (MODE_KCENTER, MODE_PPAL)


def _panel_config() -> foo.PanelConfig:
    cfg = foo.PanelConfig(
        name=PANEL_NAME,
        label=PANEL_LABEL,
        icon="cloud_upload",
    )
    cfg.unlisted = False
    return cfg


def _default_anno_key() -> str:
    return default_anno_key()


def _parse_classes(raw: str) -> list[str]:
    classes = [part.strip() for part in (raw or "").split(",") if part.strip()]
    if not classes:
        raise ValueError("クラス名を 1 つ以上指定してください（カンマ区切り）")
    return classes


def _sync_ppal_classes_from_checkpoint(ctx, checkpoint: Optional[str] = None) -> None:
    ckpt = (checkpoint or ctx.panel.state.ppal_checkpoint or "").strip()
    if not ckpt:
        return
    try:
        names = read_checkpoint_classes(ckpt)
    except ValueError:
        return
    ctx.panel.state.classes = classes_csv(names)
    ctx.panel.state.ppal_allowed_classes = classes_csv(names)


def _resolve_cvat_classes(ctx) -> list[str]:
    mode = ctx.panel.state.selection_mode or MODE_KCENTER
    if mode == MODE_PPAL:
        checkpoint = normalize_ppal_checkpoint(ctx.panel.state.ppal_checkpoint)
        return list(read_checkpoint_classes(checkpoint))
    return _parse_classes(ctx.panel.state.classes)


def _embedding_model_dropdown() -> types.DropdownView:
    dropdown = types.DropdownView()
    for option in EMBEDDING_MODEL_OPTIONS:
        dropdown.add_choice(
            option["name"],
            label=option["label"],
            description=option["description"],
        )
    return dropdown


def _selection_mode_dropdown() -> types.DropdownView:
    dropdown = types.DropdownView()
    dropdown.add_choice(MODE_KCENTER, label="k-center（embedding 多様性）")
    dropdown.add_choice(
        MODE_PPAL,
        label="PPAL（不確実性 + 多様性 / RetinaNet checkpoint）",
    )
    return dropdown


def _ppal_checkpoint_dropdown() -> types.DropdownView:
    dropdown = types.DropdownView()
    options = list_ppal_checkpoint_options()
    if not options:
        dropdown.add_choice(
            "",
            label="（third_party/PPAL/work_dirs に checkpoint がありません）",
        )
        return dropdown
    for option in options:
        dropdown.add_choice(
            option["name"],
            label=option["label"],
            description=option["name"],
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


def _rank_field_for_mode(mode: str) -> str:
    if mode == MODE_PPAL:
        return PPAL_RANK_FIELD
    return RANK_FIELD


def _sync_cvat_sent_tags(ctx) -> None:
    sync_cvat_sent_tags(ctx.dataset)
    ctx.panel.state.cvat_sent_status = cvat_sent_status_message(ctx.dataset)


class CvatKCenterPanel(foo.Panel):
    @property
    def config(self):
        return _panel_config()

    def on_load(self, ctx):
        ctx.panel.state.anno_key = ctx.panel.get_state("anno_key", _default_anno_key())
        ctx.panel.state.label_field = ctx.panel.get_state("label_field", "ground_truth")
        ctx.panel.state.classes = ctx.panel.get_state("classes", "person,car,dog")
        default_mode = os.environ.get("MMPLATFORM_SELECTION_MODE", MODE_KCENTER)
        if default_mode not in SELECTION_MODES:
            default_mode = MODE_KCENTER
        ctx.panel.state.selection_mode = ctx.panel.get_state(
            "selection_mode", default_mode
        )
        ctx.panel.state.embedding_model = ctx.panel.get_state(
            "embedding_model", DEFAULT_EMBEDDING_MODEL
        )
        checkpoint_options = list_ppal_checkpoint_options()
        default_checkpoint = (
            checkpoint_options[0]["name"] if checkpoint_options else ""
        )
        ctx.panel.state.ppal_checkpoint = ctx.panel.get_state(
            "ppal_checkpoint", default_checkpoint
        )
        _sync_ppal_classes_from_checkpoint(ctx, default_checkpoint)
        ctx.panel.state.status = ctx.panel.get_state(
            "status",
            "選定モードを選び、並べ替えてから CVAT に送れます。",
        )
        ctx.panel.state.kcenter_running = ctx.panel.get_state("kcenter_running", False)
        ctx.panel.state.kcenter_progress = ctx.panel.get_state("kcenter_progress", 0.0)
        ctx.panel.state.kcenter_progress_label = ctx.panel.get_state(
            "kcenter_progress_label", ""
        )
        ctx.panel.state.kcenter_log = ctx.panel.get_state("kcenter_log", "")
        ctx.panel.state.show_kcenter_log = ctx.panel.get_state("show_kcenter_log", False)
        _sync_cvat_sent_tags(ctx)

    def _progress_hooks(self, ctx) -> KCenterProgress:
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

    def on_selection_mode_change(self, ctx):
        value = ctx.params.get("value")
        if value in SELECTION_MODES:
            ctx.panel.state.selection_mode = value
            if value == MODE_PPAL:
                _sync_ppal_classes_from_checkpoint(ctx)

    def on_embedding_model_change(self, ctx):
        value = ctx.params.get("value")
        if value:
            ctx.panel.state.embedding_model = value

    def on_ppal_checkpoint_change(self, ctx):
        value = ctx.params.get("value")
        if value:
            ctx.panel.state.ppal_checkpoint = value
            _sync_ppal_classes_from_checkpoint(ctx, value)

    def _target_view(self, ctx):
        if ctx.view != ctx.dataset.view():
            return ctx.view
        return ctx.dataset

    def _run_kcenter(self, ctx, target, progress: KCenterProgress) -> None:
        model_name = normalize_embedding_model_name(ctx.panel.state.embedding_model)
        count = target.count()
        progress.log(f"モード: k-center")
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

    def _run_ppal(self, ctx, target, progress: KCenterProgress) -> None:
        _sync_cvat_sent_tags(ctx)
        checkpoint = normalize_ppal_checkpoint(ctx.panel.state.ppal_checkpoint)
        count = target.count()
        progress.log("モード: PPAL")
        progress.log(f"checkpoint: {checkpoint}")
        progress.log(f"View 内サンプル数: {count}")

        label_field = (ctx.panel.state.label_field or "ground_truth").strip()
        result = compute_ppal_order(
            target,
            checkpoint,
            budget=DEFAULT_SELECT,
            label_field=label_field,
            progress=progress,
        )
        progress.set(0.98, "ビューを更新中…")
        set_ppal_ranks(ctx.dataset, result.ordered_ids)
        sorted_view = target.sort_by(PPAL_RANK_FIELD)
        ctx.ops.set_view(sorted_view)

        default_n = min(DEFAULT_SELECT, result.unsent_count)
        ctx.ops.set_selected_samples(result.ordered_ids[:default_n])

        sent_note = (
            f"、送信済み {result.sent_count} 枚は末尾"
            if result.sent_count
            else ""
        )
        summary = (
            f"未送信 {result.unsent_count} 枚を PPAL 順に並べ替え、"
            f"先頭 {default_n} 枚を選択しました{sent_note}。"
            f"（DCUS pool: {result.pool_size}, checkpoint: {checkpoint}）"
        )
        progress.set(1.0, "完了", force=True)
        progress.log(summary)
        ctx.panel.state.kcenter_running = False
        ctx.panel.state.status = summary
        ctx.ops.set_progress(label="PPAL 完了", progress=1.0)
        ctx.ops.notify(summary, variant="success")

    def order_and_select(self, ctx):
        target = self._target_view(ctx)
        count = target.count()
        if count == 0:
            ctx.ops.notify("サンプルがありません", variant="error")
            return

        mode = ctx.panel.state.selection_mode or MODE_KCENTER
        ctx.panel.state.kcenter_running = True
        ctx.panel.state.kcenter_progress = 0.0
        ctx.panel.state.kcenter_progress_label = "選定処理を開始…"
        ctx.panel.state.kcenter_log = ""
        ctx.panel.state.show_kcenter_log = False

        if mode == MODE_KCENTER:
            ctx.panel.state.status = "k-center 処理を開始…"
        else:
            ctx.panel.state.status = "PPAL 処理を開始…"

        progress = self._progress_hooks(ctx)
        try:
            if mode == MODE_PPAL:
                self._run_ppal(ctx, target, progress)
            else:
                try:
                    normalize_embedding_model_name(ctx.panel.state.embedding_model)
                except ValueError as exc:
                    raise ValueError(str(exc)) from exc
                self._run_kcenter(ctx, target, progress)
        except Exception as exc:
            progress.log(f"ERROR: {exc}")
            ctx.panel.state.kcenter_running = False
            ctx.panel.state.status = f"失敗: {exc}"
            ctx.ops.set_progress(label=f"失敗: {exc}", progress=0.0)
            ctx.ops.notify(str(exc), variant="error")

    def select_default(self, ctx):
        target = self._target_view(ctx)
        mode = ctx.panel.state.selection_mode or MODE_KCENTER
        rank_field = _rank_field_for_mode(mode)
        sorted_view = target.sort_by(rank_field)
        if mode == MODE_PPAL:
            _sync_cvat_sent_tags(ctx)
            sorted_view = sorted_view.match_tags(CVAT_SENT_TAG, bool=False)
        ids = [sample.id for sample in sorted_view.limit(DEFAULT_SELECT)]
        if not ids:
            ctx.ops.notify(
                f"順位が未計算か、未送信画像がありません。先に「並べ替え」を実行してください。（{mode}）",
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
        try:
            classes = _resolve_cvat_classes(ctx)
        except ValueError as exc:
            ctx.ops.notify(str(exc), variant="error")
            ctx.panel.state.status = str(exc)
            return

        ctx.panel.state.status = f"CVAT へ {len(selected)} 枚を送信中… ({cvat_env_summary()})"
        try:
            used_key = send_samples_to_cvat(
                ctx.dataset,
                selected,
                anno_key=anno_key,
                label_field=label_field,
                classes=classes,
                launch_editor=True,
            )
            ctx.panel.state.anno_key = _default_anno_key()
            _sync_cvat_sent_tags(ctx)
            ctx.panel.state.status = (
                f"CVAT 送信完了: anno_key='{used_key}', {len(selected)} 枚。"
                f" {ctx.panel.state.cvat_sent_status}"
            )
            ctx.ops.notify(ctx.panel.state.status, variant="success")
        except Exception as exc:
            ctx.panel.state.anno_key = _default_anno_key()
            ctx.panel.state.status = f"CVAT 送信失敗: {exc}"
            ctx.ops.notify(str(exc), variant="error")

    def render(self, ctx):
        panel = types.Object()
        mode = ctx.panel.state.selection_mode or MODE_KCENTER
        ppal_options = list_ppal_checkpoint_options()

        panel.md(
            f"""
### CVAT 能動学習選定

1. **選定モード**（k-center または PPAL）を選択
2. **並べ替え** — 未送信画像を PPAL 順に並べ、送信済み（`{CVAT_SENT_TAG}`）は末尾へ。先頭 {DEFAULT_SELECT} 枚（未送信）を選択
3. グリッドで {MIN_SELECT} 枚以上を選択
4. **CVAT に送信**

{cvat_env_summary()}

送信済み画像は annotation run 履歴から自動でタグ `{CVAT_SENT_TAG}` が付きます。
App のフィルタで `tags` = `{CVAT_SENT_TAG}` とすると送信済みのみ表示できます。

{ctx.panel.state.cvat_sent_status or cvat_sent_status_message(ctx.dataset)}

{_python_version_note()}

{selection_bounds_message(len(ctx.selected or []))}
            """.strip(),
            name="intro",
        )

        panel.str(
            "selection_mode",
            label="選定モード",
            view=_selection_mode_dropdown(),
            on_change=self.on_selection_mode_change,
            required=True,
        )

        if mode == MODE_KCENTER:
            panel.str(
                "embedding_model",
                label="Embedding model",
                view=_embedding_model_dropdown(),
                on_change=self.on_embedding_model_change,
                required=True,
            )
        else:
            panel.str(
                "ppal_checkpoint",
                label="PPAL checkpoint（third_party/PPAL/work_dirs）",
                view=_ppal_checkpoint_dropdown(),
                on_change=self.on_ppal_checkpoint_change,
                required=True,
            )
            panel.md(
                "画像のみのデータセットでも利用できます（COCO アノテーション不要）。"
                " 初回は metadata を自動計算します。",
                name="ppal_image_only_help",
            )
            if not ppal_options:
                panel.md(
                    "PPAL checkpoint がありません。"
                    " `third_party/PPAL/work_dirs` に **PPAL 学習済み**"
                    " RetinaNet の `.pth` を配置してください"
                    "（`bbox_head.class_quality` 必須）。",
                    name="ppal_checkpoint_help",
                )

        panel.str("anno_key", label="Annotation key", required=True)
        panel.str("label_field", label="Label field", required=True)
        if mode == MODE_PPAL:
            allowed = (ctx.panel.state.ppal_allowed_classes or ctx.panel.state.classes or "").strip()
            if not allowed and ctx.panel.state.ppal_checkpoint:
                _sync_ppal_classes_from_checkpoint(ctx)
                allowed = (ctx.panel.state.classes or "").strip()
            panel.str(
                "classes",
                label="Classes（checkpoint 固定）",
                description="PPAL checkpoint の学習クラスと一致します。変更できません。",
                required=True,
                read_only=True,
            )
        else:
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

        order_label = "PPAL で並べ替え" if mode == MODE_PPAL else "k-center で並べ替え"
        panel.btn(
            "order_btn",
            label=order_label,
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

        used_key = send_samples_to_cvat(
            ctx.dataset,
            selected,
            anno_key=anno_key,
            label_field=label_field,
            classes=classes,
            launch_editor=True,
        )
        status = cvat_sent_status_message(ctx.dataset)

        return {
            "message": (
                f"Sent {len(selected)} samples to CVAT (anno_key={used_key}). {status}"
            )
        }


class OpenCvatKcenterPanel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="open_cvat_kcenter_panel",
            label="Open CVAT active-learning panel",
            description="Open the CVAT k-center / PPAL selection panel in the App",
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
