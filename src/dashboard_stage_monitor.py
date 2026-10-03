# KAMPact - dashboard stage monitor
# 11_realtime_dashboard.py 가 import 하는 보조 모듈입니다. (src/ 폴더에 같이 두세요)
#
#  - run_command_live : 하위 스크립트를 실행하며 stdout/stderr 를 실시간으로 흘려줌
#  - parse_progress   : 로그 한 줄에서 진행률(37%, 3/5 ...) 추출
#  - stage_details    : 각 단계가 만든 실제 산출물(csv)을 읽어 결과 요약 생성
#  - TimingStore      : 단계별 소요 시간을 저장 -> 다음 실행 때 예상 진행률/잔여 시간 계산
#  - StageBoard       : 단계 실행 현황판 / 현재 단계 / 로그 / 결과 상세 UI
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
import unicodedata
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

STATUS_TXT = {
    "pending": "대기",
    "running": "실행 중",
    "done": "완료",
    "reused": "재사용",
    "failed": "실패",
    "skipped": "건너뜀",
}


# ---------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------
def _cw(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def vlen(s: str) -> int:
    return sum(_cw(c) for c in s)


def clip(s: str, width: int) -> str:
    """화면 폭(한글=2칸) 기준으로 문자열 자르기."""
    s = str(s).replace("\t", " ").rstrip()
    if vlen(s) <= width:
        return s
    out, n = [], 0
    for c in s:
        w = _cw(c)
        if n + w > width - 1:
            break
        out.append(c)
        n += w
    return "".join(out) + "…"


def fmt_dur(sec) -> str:
    if sec is None:
        return "-"
    m, s = divmod(float(sec), 60.0)
    return f"{int(m):02d}:{s:04.1f}"


_PCT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%")
_FRAC = re.compile(r"(?<![\d/.\-:])(\d+)\s*/\s*(\d+)(?![\d/.\-:])")


def parse_progress(line: str):
    """'45%|####', 'fold 3/5', '[2/10]' 같은 로그에서 0~1 진행률을 뽑는다. 없으면 None."""
    m = _PCT.search(line)
    if m:
        v = float(m.group(1))
        if 0.0 <= v <= 100.0:
            return v / 100.0
    hits = _FRAC.findall(line)
    if hits:
        cur, tot = map(int, hits[-1])
        if tot >= 2 and 0 <= cur <= tot:
            return cur / tot
    return None


# ---------------------------------------------------------------------
# live subprocess
# ---------------------------------------------------------------------
def run_command_live(root, cmd, env=None, on_update=None, wait=None, poll=0.1):
    """cmd 를 실행하고 출력 줄을 실시간으로 on_update(new_lines, all_lines) 에 전달.

    wait(sec) 에 plt.pause 를 넘기면 대기 시간 동안 GUI 이벤트도 같이 처리된다.
    returns (returncode, all_lines)
    """
    merged = os.environ.copy()
    merged["PYTHONUNBUFFERED"] = "1"
    merged["PYTHONIOENCODING"] = "utf-8"
    if env:
        merged.update(env)

    proc = subprocess.Popen(
        cmd, cwd=str(root), env=merged,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    q: "queue.Queue[str | None]" = queue.Queue()

    def reader():
        try:
            for line in proc.stdout:
                q.put(line.rstrip("\r\n"))
        finally:
            q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    wait = wait or time.sleep
    lines: list[str] = []
    finished = False
    try:
        while not finished:
            new: list[str] = []
            try:
                while True:
                    item = q.get_nowait()
                    if item is None:
                        finished = True
                        break
                    new.append(item)
            except queue.Empty:
                pass
            lines.extend(new)
            if on_update:
                on_update(new, lines)
            if not finished:
                wait(poll)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    return proc.returncode, lines


class TimingStore:
    """단계별 소요 시간 저장소 (json)."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}

    def get(self, stage_no):
        v = self.data.get(str(stage_no))
        return float(v) if v else None

    def put(self, stage_no, seconds):
        self.data[str(stage_no)] = round(float(seconds), 2)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------
# stage result digests (실제 산출물을 읽어서 요약)
# ---------------------------------------------------------------------
_METRIC_KEYS = ("f1", "recall", "precision", "detection", "delay", "auc", "false", "fp", "fn")
_SHORT = (("event_detection_rate", "event"), ("delay_mean_sec", "delay_mean"),
          ("precision", "prec"), ("_mean", ""))


def _short(col: str) -> str:
    out = col
    for a, b in _SHORT:
        out = out.replace(a, b)
    return out.strip("_") or col


def _count_rows(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return max(0, n - 1)


def csv_digest(path, max_rows: int = 4, max_metrics: int = 3) -> list[str]:
    path = Path(path)
    if not path.exists():
        return [f"{path.name}  (파일 없음)"]
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return [f"{path.name}  읽기 실패: {exc}"]
    out = [f"{path.name}  {len(df):,}행 x {len(df.columns)}열"]
    num = [c for c in df.columns
           if pd.api.types.is_numeric_dtype(df[c]) and any(k in c.lower() for k in _METRIC_KEYS)]
    num.sort(key=lambda c: (not c.endswith("_mean"), c))
    num = num[:max_metrics]
    if not num or df.empty:
        return out
    id_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    idc = id_cols[0] if id_cols else None
    key = next((c for k in ("f1", "event", "recall") for c in num if k in c.lower()), None)
    view = df.sort_values(key, ascending=False) if key else df
    for pos, (_, r) in enumerate(view.head(max_rows).iterrows()):
        tag = str(r[idc]) if idc else f"#{pos + 1}"
        vals = "  ".join(f"{_short(c)} {r[c]:.4g}" for c in num)
        out.append(f"  {tag}: {vals}")
    return out


def _label_counts(path: Path) -> list[str]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    for c in ("label", "y", "is_fault", "ground_truth", "target"):
        if c in df.columns and df[c].nunique() <= 5:
            vc = df[c].value_counts().sort_index()
            return ["  " + c + ": " + ", ".join(f"{k}={v:,}" for k, v in vc.items())]
    return []


def _stage_details(root: Path, no: int) -> list[str]:
    if no == 1:
        out = []
        for rel in ("data/press_data_normal.csv", "data/outlier_data.csv"):
            p = root / rel
            out.append(f"{p.name}  {_count_rows(p):,}행" if p.exists() else f"{p.name}  (없음)")
        return out
    if no == 2:
        p = root / "data/press_data_normal_with_idle.csv"
        df = pd.read_csv(p, usecols=["TimeStamp", "Idle"])
        idle = df[pd.to_numeric(df["Idle"], errors="coerce").fillna(0).eq(1)]
        out = [f"전체 {len(df):,}행  |  Idle {len(idle):,}행 ({len(idle) / max(len(df), 1) * 100:.1f}%)"]
        if len(idle):
            out.append(f"Idle 구간  {idle['TimeStamp'].iloc[0]}")
            out.append(f"        ~  {idle['TimeStamp'].iloc[-1]}")
        return out
    if no == 3:
        d = root / "result/time_structure_analysis"
        files = sorted(d.glob("*.csv")) if d.exists() else []
        out = [f"산출물 csv {len(files)}개"]
        out += [f"  {f.name}  {_count_rows(f):,}행" for f in files[:8]]
        return out
    if no in (4, 5):
        sub = "1.0_0.1" if no == 4 else "0.5_0.1"
        p = root / f"result/modeling_dataset_{sub}/model_windows.csv"
        return csv_digest(p)[:1] + _label_counts(p)
    table = {
        6: "outputs/5_2_mahalanobis_cv/cv_comparison.csv",
        7: "outputs/5_3_sample_level_cv/sample_cv_comparison.csv",
        8: "outputs/6_three_detector_ensemble/ensemble_summary.csv",
        9: "outputs/7_isolation_forest_baseline/summary.csv",
        10: "outputs/8_fp_fn_analysis/fn_event_summary.csv",
    }
    if no in table:
        out = csv_digest(root / table[no], max_rows=5)
        if no == 10:
            p = root / "outputs/8_fp_fn_analysis/fp_windows_1.0s.csv"
            if p.exists():
                out.append(f"fp_windows_1.0s.csv  {_count_rows(p):,}행")
        return out
    if no == 11:
        out = csv_digest(root / "outputs/10_variable_effect_analysis/variable_effect_summary.csv", max_rows=5)
        d = root / "outputs/9_fn_visualization"
        if d.exists():
            out.append(f"FN 시각화 이미지  {len(list(d.glob('*.png')))}개")
        return out
    return []


def stage_details(root, no: int) -> list[str]:
    try:
        return _stage_details(Path(root), no)
    except Exception as exc:
        return [f"상세 결과 읽기 실패: {exc}"]


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
class StageBoard:
    BAR_X, BAR_W = 0.025, 0.43

    def __init__(self, fig, ax_board, panels, labels, colors, font=None, mono=None,
                 timing=None, overall_bar=None, overall_text=None, pipe_cb=None,
                 n_progress=None, log_rows=17):
        """panels = {"cur": ax, "log": ax, "detail": ax}  (기존 카드 axes 를 단계 실행 중에만 재사용)"""
        self.fig, self.ax, self.C = fig, ax_board, colors
        self.font = font or "DejaVu Sans"
        self.mono = mono or "DejaVu Sans Mono"
        self.labels = list(labels)
        self.n = len(self.labels)
        self.n_prog = n_progress or self.n
        self.timing, self.overall_bar, self.overall_text = timing, overall_bar, overall_text
        self.pipe_cb, self.log_rows = pipe_cb, log_rows
        self.t_start = time.time()
        self.current = None
        self.stream_mode = False
        self.detail_failed = False
        self.detail_lines: list[str] = ["완료된 단계의 상세 결과가 여기에 표시됩니다."]
        self.log_tail: deque[str] = deque(maxlen=400)
        self._last = 0.0
        self.st = [
            dict(status="pending", t0=None, dur=None, log_frac=0.0, parts=[], script="",
                 lines=0, last="", expected=(timing.get(i + 1) if timing else None))
            for i in range(self.n)
        ]
        self._build_board()
        self._build_panels(panels)
        self.render(force=True)

    # ----- construction -------------------------------------------------
    def _build_board(self):
        C, ax, T = self.C, self.ax, self.ax.transAxes
        ax.set_facecolor(C["panel"])
        ax.set_zorder(10)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(C["border"])
            sp.set_linewidth(1.1)
        ax.set_title("파이프라인 실행 현황", loc="left", fontsize=10.5, fontweight="bold",
                     color=C["text"], pad=7, fontname=self.font)
        ax.text(1.0, 1.015, "실제 스크립트 실행  ·  로그 기반 진행률  ·  완료 후 결과 요약",
                transform=T, ha="right", va="bottom", fontsize=8, color=C["muted"], fontname=self.font)
        for x, t in ((0.025, "단계"), (0.285, "상태"), (0.375, "경과 / 이전 실행"), (0.52, "지금 하는 일 / 결과 요약")):
            ax.text(x, 0.962, t, transform=T, ha="left", va="center", fontsize=8,
                    color=C["muted"], fontname=self.font)
        ax.plot([0.01, 0.99], [0.940, 0.940], transform=T, color=C["dim"], lw=0.9)

        top, bottom = 0.930, 0.012
        step = (top - bottom) / self.n
        self.rows = []
        for i, label in enumerate(self.labels):
            yc = top - (i + 0.5) * step
            strip = Rectangle((0.006, yc - step * 0.40), 0.007, step * 0.80, transform=T,
                              facecolor=C["dim"], edgecolor="none")
            ax.add_patch(strip)
            name = ax.text(0.025, yc + step * 0.18, label, transform=T, ha="left", va="center",
                           fontsize=9.5, fontweight="bold", color=C["muted"], fontname=self.font)
            status = ax.text(0.285, yc + step * 0.18, "", transform=T, ha="left", va="center",
                             fontsize=8.8, fontweight="bold", color=C["muted"], fontname=self.font)
            dur = ax.text(0.375, yc + step * 0.18, "", transform=T, ha="left", va="center",
                          fontsize=8.2, color=C["muted"], family=self.mono)
            bar_bg = Rectangle((self.BAR_X, yc - step * 0.34), self.BAR_W, step * 0.15, transform=T,
                               facecolor=C["bg"], edgecolor=C["border"], linewidth=0.6)
            bar_fill = Rectangle((self.BAR_X, yc - step * 0.34), 0, step * 0.15, transform=T,
                                 facecolor=C["dim"], edgecolor="none")
            ax.add_patch(bar_bg)
            ax.add_patch(bar_fill)
            parts = ax.text(0.52, yc, "", transform=T, ha="left", va="center", fontsize=8,
                            color=C["muted"], fontname=self.font, linespacing=1.35)
            if i < self.n - 1:
                ax.plot([0.01, 0.99], [yc - step / 2, yc - step / 2], transform=T,
                        color=C["grid"], lw=0.5)
            self.rows.append(dict(strip=strip, name=name, status=status, dur=dur,
                                  bar_fill=bar_fill, parts=parts))

    def _build_panels(self, panels):
        C = self.C
        self._orig = {}
        stage_titles = {"cur": ("현재 실행 단계", "진행 상황"),
                        "log": ("실행 로그", "스크립트 출력 (최근)"),
                        "detail": ("단계 결과 상세", "마지막 완료 단계")}
        for key, ax in panels.items():
            sub = next((t for t in ax.texts
                        if abs(t.get_position()[0] - 1.0) < 1e-6 and abs(t.get_position()[1] - 1.015) < 1e-6), None)
            self._orig[key] = (ax, ax.get_title(loc="left"), sub, sub.get_text() if sub else "")
            ax.set_title(stage_titles[key][0], loc="left")
            if sub:
                sub.set_text(stage_titles[key][1])

        ax = panels["cur"]
        T = ax.transAxes
        self.cur_text = ax.text(0.05, 0.94, "", transform=T, va="top", ha="left", fontsize=8.4,
                                family=self.mono, color=C["text"], linespacing=1.45)
        self.cur_bar_bg = Rectangle((0.05, 0.47), 0.90, 0.05, transform=T, facecolor=C["bg"],
                                    edgecolor=C["border"], linewidth=0.6)
        self.cur_bar_fill = Rectangle((0.05, 0.47), 0, 0.05, transform=T, facecolor=C["amber"],
                                      edgecolor="none")
        ax.add_patch(self.cur_bar_bg)
        ax.add_patch(self.cur_bar_fill)
        self.cur_last = ax.text(0.05, 0.41, "", transform=T, va="top", ha="left", fontsize=7.6,
                                family=self.mono, color=C["muted"], linespacing=1.4)
        self.log_text = panels["log"].text(0.04, 0.965, "", transform=panels["log"].transAxes, va="top",
                                           ha="left", fontsize=7.4, family=self.mono, color=C["text"],
                                           linespacing=1.38)
        self.detail_text = panels["detail"].text(0.05, 0.965, "", transform=panels["detail"].transAxes,
                                                 va="top", ha="left", fontsize=7.9, family=self.mono,
                                                 color=C["text"], linespacing=1.45)
        self._panel_artists = [self.cur_text, self.cur_bar_bg, self.cur_bar_fill, self.cur_last,
                               self.log_text, self.detail_text]

    # ----- state transitions --------------------------------------------
    def _pump(self, sec: float = 0.02):
        if self.stream_mode:
            return
        try:
            plt.pause(sec)
        except Exception:
            time.sleep(sec)

    def begin(self, idx: int, script: str = "", note: str = ""):
        s = self.st[idx]
        if s["status"] != "running":
            s.update(status="running", t0=time.time(), dur=None, log_frac=0.0, parts=[],
                     lines=0, last="", script=script)
            self.log_tail.append(f"=== [{idx + 1}] {self.labels[idx]} ===")
        elif script:
            s["script"] = script
        self.current = idx
        if note:
            self.note(note)
        self.render(force=True)
        self._pump()

    def note(self, text: str):
        self.log_tail.append(text)
        if self.current is not None:
            self.st[self.current]["last"] = text
        self.render()

    def feed(self, new_lines, all_lines=None):
        """run_command_live 의 on_update 콜백. 새 줄이 없어도 호출되어 경과 시간이 갱신된다."""
        if self.current is not None:
            s = self.st[self.current]
            for line in new_lines:
                # tqdm 막대(블록 문자) 등 한글 폰트에 없는 글자는 '#' 로 치환 (경고 방지)
                line = re.sub("[\u2580-\u259f]", "#", line).strip()
                if not line:
                    continue
                s["lines"] += 1
                s["last"] = line
                self.log_tail.append(line)
                p = parse_progress(line)
                if p is not None:
                    s["log_frac"] = max(s["log_frac"], p)
        self.render()

    def _close(self, idx, status, parts):
        s = self.st[idx]
        if s["t0"] is not None:
            s["dur"] = time.time() - s["t0"]
        s["status"], s["parts"] = status, list(parts or [])
        if self.current == idx:
            self.current = None

    def finish(self, idx: int, parts=None, details=None, status: str = "done"):
        self._close(idx, status, parts)
        s = self.st[idx]
        if status == "done" and s["dur"] is not None and self.timing:
            self.timing.put(idx + 1, s["dur"])
        dur = f"  ({fmt_dur(s['dur'])})" if s["dur"] is not None else ""
        self.detail_failed = False
        self.detail_lines = [f"[{idx + 1}] {self.labels[idx]}  {STATUS_TXT[status]}{dur}", "-" * 34]
        self.detail_lines += list(s["parts"]) + [""] + list(details or [])
        self.render(force=True)
        self._pump()

    def fail(self, idx: int, message: str, tail=None):
        self._close(idx, "failed", [message])
        self.detail_failed = True
        self.detail_lines = [f"[{idx + 1}] {self.labels[idx]}  실패", "-" * 34, message, ""] + list(tail or [])
        self.render(force=True)
        self._pump()

    def skip(self, idx: int, message: str = ""):
        self._close(idx, "skipped", [message] if message else [])
        self.render(force=True)
        self._pump()

    def run_with_pump(self, fn):
        """fn 을 별도 스레드에서 돌리고 기다리는 동안 UI 를 계속 갱신 (모델 학습 등 오래 걸리는 계산용)."""
        box = {}

        def work():
            try:
                box["result"] = fn()
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc

        th = threading.Thread(target=work, daemon=True)
        th.start()
        while th.is_alive():
            self.render()
            self._pump(0.08)
        if "error" in box:
            raise box["error"]
        return box.get("result")

    # ----- rendering ----------------------------------------------------
    def _color(self, status: str):
        C = self.C
        return {"running": C["amber"], "done": C["green"], "reused": C["cyan"],
                "failed": C["red"], "skipped": C["muted"]}.get(status, C["dim"])

    @staticmethod
    def _frac(s, elapsed):
        st = s["status"]
        if st == "running":
            if s["log_frac"] > 0:
                return s["log_frac"], "log"
            if s["expected"] and elapsed is not None:
                return min(0.95, elapsed / s["expected"]), "eta"
            return None, "none"
        if st in ("done", "reused", "failed"):
            return 1.0, ""
        return 0.0, ""

    @staticmethod
    def _wrap_parts(parts, width=46, max_lines=2):
        lines, cur = [], ""
        for p in parts:
            p = str(p).strip()
            if not p:
                continue
            cand = p if not cur else cur + "  ·  " + p
            if cur and vlen(cand) > width:
                lines.append(cur)
                cur = p
            else:
                cur = cand
        if cur:
            lines.append(cur)
        lines = lines[:max_lines]
        return [clip(x, width) for x in lines]

    def pipeline_summary_lines(self):
        sts = [s["status"] for s in self.st[:self.n_prog]]
        total = sum(s["dur"] or 0 for s in self.st[:self.n_prog])
        out = [f"완료 {sts.count('done')}  재사용 {sts.count('reused')}  건너뜀 {sts.count('skipped')}  "
               f"실패 {sts.count('failed')}  대기 {sts.count('pending')}",
               f"단계 소요 합계  {fmt_dur(total)}"]
        timed = [(s["dur"], i) for i, s in enumerate(self.st[:self.n_prog]) if s["dur"]]
        if timed:
            d, i = max(timed)
            out.append(f"가장 오래 걸린 단계  {i + 1} ({fmt_dur(d)})")
        return out

    def render(self, force: bool = False):
        # 스트리밍 중에는 현황판이 보일 때('p' 키)만 갱신한다.
        if self.stream_mode and not self.ax.get_visible():
            return
        now = time.time()
        if not force and now - self._last < 0.1:
            return
        self._last = now
        C = self.C
        spin = "|/-\\"[int(now * 6) % 4]
        bounce = abs(((now * 0.8) % 2.0) - 1.0)
        statuses, shorts = [], []
        cur_frac, done_cnt = 0.0, 0

        for i, s in enumerate(self.st):
            row, stt = self.rows[i], s["status"]
            col = self._color(stt)
            elapsed = (now - s["t0"]) if (stt == "running" and s["t0"]) else None
            exp = s["expected"]
            frac, src = self._frac(s, elapsed)

            row["strip"].set_facecolor(col)
            row["name"].set_color(C["muted"] if stt == "pending" else C["text"])
            stxt = STATUS_TXT[stt]
            if stt == "running":
                if frac is not None:   # 로그 기준 = 실제 진행률, '~' = 이전 실행 시간 기반 추정
                    stxt += f" {spin} " + ("" if src == "log" else "~") + f"{int(frac * 100)}%"
                else:
                    stxt += f" {spin}"
            row["status"].set_text(stxt)
            row["status"].set_color(col)

            if stt == "running":
                dtxt = fmt_dur(elapsed) + (f" / ~{fmt_dur(exp)}" if exp else "")
            elif stt in ("done", "failed") and s["dur"] is not None:
                dtxt = fmt_dur(s["dur"]) + (f" / ~{fmt_dur(exp)}" if exp and stt == "done" else "")
            elif stt == "pending" and exp:
                dtxt = f"예상 ~{fmt_dur(exp)}"
            else:
                dtxt = ""
            row["dur"].set_text(dtxt)

            bar = row["bar_fill"]
            bar.set_facecolor(col)
            if stt == "running" and frac is None:
                seg = 0.28 * self.BAR_W
                bar.set_x(self.BAR_X + (self.BAR_W - seg) * bounce)
                bar.set_width(seg)
            else:
                bar.set_x(self.BAR_X)
                bar.set_width(self.BAR_W * (frac or 0.0))

            if stt == "running":
                body = [clip(s["last"] or "시작 중 ...", 46)]
                row["parts"].set_color(C["amber"])
            else:
                body = self._wrap_parts(s["parts"])
                row["parts"].set_color(C["text"] if stt in ("done", "reused") else
                                       C["red"] if stt == "failed" else C["muted"])
            row["parts"].set_text("\n".join(body))

            statuses.append(stt)
            if stt == "running":
                shorts.append(fmt_dur(elapsed))
            elif stt in ("done", "failed") and s["dur"] is not None:
                shorts.append(fmt_dur(s["dur"]))
            elif stt == "reused":
                shorts.append("재사용")
            elif stt == "skipped":
                shorts.append("건너뜀")
            else:
                shorts.append("")
            if i < self.n_prog:
                if stt in ("done", "reused", "skipped"):
                    done_cnt += 1
                elif stt == "running" and frac is not None:
                    cur_frac = frac

        # overall progress (header)
        overall = min(1.0, (done_cnt + cur_frac) / self.n_prog)
        if self.overall_bar is not None:
            self.overall_bar.set_width(overall)
        if self.overall_text is not None:
            rem, unknown = 0.0, False
            for s in self.st[:self.n_prog]:
                if s["status"] == "pending":
                    if s["expected"]:
                        rem += s["expected"]
                    else:
                        unknown = True
                elif s["status"] == "running":
                    if s["expected"]:
                        rem += max(s["expected"] - (now - s["t0"]), 0.0)
                    else:
                        unknown = True
            if done_cnt >= self.n_prog:
                txt = f"전체 {self.n_prog}/{self.n_prog} 단계 완료  ·  총 {fmt_dur(now - self.t_start)}"
            else:
                eta = "-" if (rem == 0 and unknown) else f"~{fmt_dur(rem)}{'+' if unknown else ''}"
                txt = f"전체 {done_cnt}/{self.n_prog} 단계  ·  경과 {fmt_dur(now - self.t_start)}  ·  잔여 예상 {eta}"
            self.overall_text.set_text(txt)

        if self.pipe_cb:
            self.pipe_cb(statuses, shorts)

        self._render_current(now, spin, bounce)
        self.log_text.set_text("\n".join(clip(x, 54) for x in list(self.log_tail)[-self.log_rows:]))
        self.detail_text.set_text("\n".join(clip(x, 56) for x in self.detail_lines[:30]))
        self.detail_text.set_color(C["red"] if self.detail_failed else C["text"])

        try:
            self.fig.canvas.draw_idle()
            if not self.stream_mode:      # 애니메이션 콜백 안에서 이벤트 루프 재진입 방지
                self.fig.canvas.flush_events()
        except Exception:
            pass

    def _render_current(self, now, spin, bounce):
        C = self.C
        if self.current is None:
            started = any(s["status"] != "pending" for s in self.st)
            self.cur_text.set_text("\n".join(self.pipeline_summary_lines()) if started else "대기 중 ...")
            self.cur_bar_fill.set_width(0.0)
            self.cur_last.set_text("")
            return
        i = self.current
        s = self.st[i]
        elapsed = now - s["t0"] if s["t0"] else 0.0
        exp = s["expected"]
        frac, src = self._frac(s, elapsed)
        pct = {"log": f"{int((frac or 0) * 100)}%  [LOG] 실제 진행률",
               "eta": f"~{int((frac or 0) * 100)}%  [EST] 이전 실행 기준 추정",
               "none": "측정 불가  (로그에 진행률 없음)"}[src]
        name = self.labels[i].split(None, 1)[-1]
        lines = [
            f"단계      {i + 1}/{self.n}  {clip(name, 30)}",
            f"스크립트  {clip(s['script'] or '-', 34)}",
            f"상태      실행 중 {spin}",
            f"경과      {fmt_dur(elapsed)}" + (f"   (이전 {fmt_dur(exp)})" if exp else ""),
            f"진행      {pct}",
            f"로그      {s['lines']:,}줄 출력",
        ]
        self.cur_text.set_text("\n".join(lines))
        W = 0.90
        if frac is None:
            seg = 0.28 * W
            self.cur_bar_fill.set_x(0.05 + (W - seg) * bounce)
            self.cur_bar_fill.set_width(seg)
        else:
            self.cur_bar_fill.set_x(0.05)
            self.cur_bar_fill.set_width(W * frac)
        self.cur_last.set_text("최근 출력\n" + clip(s["last"] or "-", 50))

    # ----- mode switch / reports ---------------------------------------
    def enter_stream_mode(self):
        """파이프라인 단계가 끝나고 실시간 추론 화면으로 넘어갈 때 호출."""
        self.render(force=True)
        self.stream_mode = True
        self.ax.set_visible(False)
        for a in self._panel_artists:
            a.set_visible(False)
        for ax, title, sub, sub_text in self._orig.values():
            ax.set_title(title, loc="left")
            if sub:
                sub.set_text(sub_text)

    def toggle_board(self):
        """실시간 추론 중에도 'p' 키로 단계 실행 현황을 다시 볼 수 있다."""
        self.ax.set_visible(not self.ax.get_visible())
        if self.ax.get_visible():
            self.render(force=True)   # 다시 보여줄 때 최신 상태로 갱신
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def report_rows(self):
        return [dict(stage=i + 1, label=self.labels[i], status=s["status"],
                     seconds=None if s["dur"] is None else round(s["dur"], 2),
                     summary=" | ".join(s["parts"]))
                for i, s in enumerate(self.st)]

    def save_report(self, path):
        try:
            pd.DataFrame(self.report_rows()).to_csv(path, index=False, encoding="utf-8-sig")
            return True
        except Exception:
            return False