#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 runs/expNNN_xxx/config.yaml 直接驱动一次 KWS 实验。

它把原来需要反复编辑 sherpa_eval/run.sh / sherpa_perf/run.sh 的内容，移动到
每个实验自己的 config.yaml 中：

  python scripts/run_from_config.py runs/exp003_lights_on/config.yaml
  python scripts/run_from_config.py runs/exp003_lights_on --stage manifest,eval,parse,register,report
  python scripts/run_from_config.py runs/exp003_lights_on/config.yaml --stage sweep --only-testset car_noise
  python scripts/run_from_config.py runs/exp003_lights_on/config.yaml --stage perf --only-scene concurrent_c8

默认 stage：manifest,eval,perf,parse,register,report
显式 stage：manifest / eval / sweep / perf / parse / register / report
别名：all=默认 stage；full=manifest,eval,sweep,perf,parse,register,report

路径约定：
  - 相对路径默认按“项目根目录”解析，而不是按 config.yaml 所在目录解析。
  - 可在路径里使用 ${ROOT} / {ROOT} / ${EXP_DIR} / {EXP_DIR} 占位。

依赖：
  - 优先使用 PyYAML；若未安装，内置一个够本仓配置使用的 YAML 子集解析器。
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STAGE = ["manifest", "eval", "perf", "parse", "register", "report"]
FULL_STAGE = ["manifest", "eval", "sweep", "perf", "parse", "register", "report"]
VALID_STAGES = set(FULL_STAGE)


# ─── YAML 读取：PyYAML 优先，缺失时用仓库内置子集解析器 ──────────────────

def _strip_comment(line: str) -> str:
    """去掉不在单双引号内的 # 注释。"""
    out: List[str] = []
    quote: Optional[str] = None
    esc = False
    for ch in line:
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == "\\" and quote == '"':
            out.append(ch)
            esc = True
            continue
        if ch in ('"', "'"):
            if quote is None:
                quote = ch
            elif quote == ch:
                quote = None
            out.append(ch)
            continue
        if ch == "#" and quote is None:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _split_inline_list(s: str) -> List[str]:
    assert s.startswith("[") and s.endswith("]")
    body = s[1:-1].strip()
    if not body:
        return []
    out: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    esc = False
    for ch in body:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if ch == "\\" and quote == '"':
            buf.append(ch)
            esc = True
            continue
        if ch in ('"', "'"):
            if quote is None:
                quote = ch
            elif quote == ch:
                quote = None
            buf.append(ch)
            continue
        if ch == "," and quote is None:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf).strip())
    return out


def _coerce_scalar(v: str) -> Any:
    v = v.strip()
    if v == "":
        return ""
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    if v.startswith("[") and v.endswith("]"):
        return [_coerce_scalar(x) for x in _split_inline_list(v)]
    try:
        if re.match(r"^[+-]?\d+$", v):
            return int(v)
        if re.match(r"^[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?$", v) or re.match(r"^[+-]?\d+[eE][+-]?\d+$", v):
            return float(v)
    except ValueError:
        pass
    return v


def _split_key_value(content: str) -> Tuple[str, str]:
    """按第一个不在引号内的冒号拆 key/value。"""
    quote: Optional[str] = None
    esc = False
    for i, ch in enumerate(content):
        if esc:
            esc = False
            continue
        if ch == "\\" and quote == '"':
            esc = True
            continue
        if ch in ('"', "'"):
            if quote is None:
                quote = ch
            elif quote == ch:
                quote = None
            continue
        if ch == ":" and quote is None:
            return content[:i].strip(), content[i + 1:].strip()
    raise ValueError(f"不是 key: value 行: {content!r}")


def _preprocess_yaml(text: str) -> List[Tuple[int, str]]:
    lines: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        raw = raw.rstrip("\n")
        if not raw.strip():
            continue
        no_comment = _strip_comment(raw)
        if not no_comment.strip():
            continue
        indent = len(no_comment) - len(no_comment.lstrip(" "))
        if "\t" in no_comment[:indent]:
            raise ValueError("YAML 缩进请使用空格，不要使用 tab")
        lines.append((indent, no_comment.strip()))
    return lines


def _parse_yaml_subset(text: str) -> Dict[str, Any]:
    """足够解析本仓 config.yaml 的 YAML 子集：dict、list、标量、inline list。"""
    lines = _preprocess_yaml(text)
    if not lines:
        return {}

    def parse_block(i: int, indent: int) -> Tuple[Any, int]:
        if i >= len(lines):
            return {}, i
        cur_indent, cur_content = lines[i]
        if cur_indent < indent:
            return {}, i
        if cur_content.startswith("- "):
            return parse_list(i, cur_indent)
        return parse_dict(i, cur_indent)

    def parse_dict(i: int, indent: int) -> Tuple[Dict[str, Any], int]:
        d: Dict[str, Any] = {}
        while i < len(lines):
            cur_indent, content = lines[i]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                # 交给上一层 key 的 block 处理；这里遇到代表结构不规范。
                raise ValueError(f"缩进异常: {content!r}")
            if content.startswith("- "):
                break
            key, val = _split_key_value(content)
            if val == "":
                if i + 1 < len(lines) and lines[i + 1][0] > cur_indent:
                    child, i = parse_block(i + 1, lines[i + 1][0])
                    d[key] = child
                else:
                    d[key] = {}
                    i += 1
            else:
                d[key] = _coerce_scalar(val)
                i += 1
        return d, i

    def parse_list(i: int, indent: int) -> Tuple[List[Any], int]:
        arr: List[Any] = []
        while i < len(lines):
            cur_indent, content = lines[i]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise ValueError(f"列表缩进异常: {content!r}")
            if not content.startswith("- "):
                break
            item = content[2:].strip()
            i += 1
            if item == "":
                if i < len(lines) and lines[i][0] > cur_indent:
                    child, i = parse_block(i, lines[i][0])
                    arr.append(child)
                else:
                    arr.append(None)
                continue
            # 常见写法：- name: xxx，后续缩进键继续并入同一个 dict。
            if ":" in item and not item.startswith(('"', "'")):
                key, val = _split_key_value(item)
                obj: Dict[str, Any] = {}
                if val == "":
                    if i < len(lines) and lines[i][0] > cur_indent:
                        child, i = parse_block(i, lines[i][0])
                        obj[key] = child
                    else:
                        obj[key] = {}
                else:
                    obj[key] = _coerce_scalar(val)
                if i < len(lines) and lines[i][0] > cur_indent:
                    extra, i = parse_block(i, lines[i][0])
                    if isinstance(extra, dict):
                        obj.update(extra)
                    else:
                        raise ValueError(f"列表项 {item!r} 的后续内容必须是 dict")
                arr.append(obj)
            else:
                arr.append(_coerce_scalar(item))
                # 标量列表项不接受后续缩进块。
        return arr, i

    parsed, end = parse_block(0, lines[0][0])
    if end != len(lines):
        raise ValueError(f"YAML 解析未消费全部行: {lines[end:]}")
    if not isinstance(parsed, dict):
        raise ValueError("顶层 YAML 必须是 dict")
    return parsed


def load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return _parse_yaml_subset(text)
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} 顶层必须是 YAML dict")
    return data


# ─── 通用工具 ───────────────────────────────────────────────────────────

def _as_list(v: Any) -> List[Any]:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        sep = "," if "," in v else None
        return [x.strip() for x in v.split(sep) if x.strip()]
    return [v]


def _arg_list(v: Any) -> List[str]:
    """CLI 追加参数: list 原样；字符串按 shell 规则拆分。"""
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        return shlex.split(v)
    return [str(v)]


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.lower() not in ("", "0", "false", "no", "off", "none", "null")
    return bool(v)


def _first_present(*vals: Any) -> Any:
    for v in vals:
        if v is not None and v != "" and v != {} and v != []:
            return v
    return None


def _stage_value(v: Any) -> str:
    """用于拼 suffix/tag 的稳定字符串。"""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _display_path(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


class Runner:
    def __init__(self, config_path: Path, cfg: Dict[str, Any], python: str, dry_run: bool = False):
        self.config_path = config_path.resolve()
        self.cfg = cfg
        self.python = python
        self.dry_run = dry_run
        self.exp_dir = self._infer_exp_dir(config_path)
        self.metrics_dir = self.exp_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _infer_exp_dir(path: Path) -> Path:
        p = path.resolve()
        if p.is_dir():
            return p
        if p.name != "config.yaml":
            # 仍然允许任意 yaml，但默认把父目录视为实验目录。
            return p.parent
        return p.parent

    def path(self, value: Any, *, must: bool = False) -> Optional[str]:
        if value is None or value == "":
            if must:
                raise SystemExit("配置缺少必填路径字段")
            return None
        s = str(value)
        repl = {
            "${ROOT}": str(ROOT),
            "{ROOT}": str(ROOT),
            "${EXP_DIR}": str(self.exp_dir),
            "{EXP_DIR}": str(self.exp_dir),
        }
        for k, v in repl.items():
            s = s.replace(k, v)
        s = os.path.expandvars(os.path.expanduser(s))
        p = Path(s)
        if not p.is_absolute():
            p = ROOT / p
        return str(p.resolve())

    def run(self, cmd: Sequence[str]) -> None:
        print("+ " + shlex.join(str(x) for x in cmd))
        if self.dry_run:
            return
        subprocess.run([str(x) for x in cmd], cwd=str(ROOT), check=True)

    def _require_model(self) -> Dict[str, str]:
        model = self.cfg.get("model") or {}
        # 兼容少量可能的键名写法。
        aliases = {
            "tokens": ["tokens"],
            "encoder": ["encoder"],
            "decoder": ["decoder"],
            "joiner": ["joiner"],
            "keywords_file": ["keywords_file", "keywords-file", "keywords"],
        }
        out: Dict[str, str] = {}
        missing: List[str] = []
        for canonical, names in aliases.items():
            val = None
            for n in names:
                val = model.get(n)
                if val not in (None, ""):
                    break
            if val in (None, ""):
                missing.append(canonical)
            else:
                out[canonical] = self.path(val, must=True) or ""
        if missing:
            raise SystemExit(
                "config.yaml 的 model 段缺少字段: " + ", ".join(missing) +
                "。如果只想 parse/register/report，请用 --stage parse,register,report。"
            )
        return out

    def _common_eval_values(self) -> Dict[str, Any]:
        ev = self.cfg.get("eval") or {}
        dec = self.cfg.get("decode") or {}
        return {
            "suffix": _first_present(ev.get("suffix"), "onnx"),
            "provider": _first_present(ev.get("provider"), "cpu"),
            "num_threads": _first_present(ev.get("num_threads"), 2),
            "chunk_seconds": _first_present(ev.get("chunk_seconds"), dec.get("chunk_seconds"), 0.5),
            "keywords_threshold": _first_present(ev.get("keywords_threshold"), dec.get("keywords_threshold")),
            "keywords_score": _first_present(ev.get("keywords_score"), dec.get("keywords_score")),
            "num_trailing_blanks": ev.get("num_trailing_blanks"),
            "max_active_paths": ev.get("max_active_paths"),
            "limit": ev.get("limit"),
            "test_only_keywords": ev.get("test_only_keywords"),
            "phrase_space_to_underscore": ev.get("phrase_space_to_underscore", True),
            "debug_lists": ev.get("debug_lists", True),
        }

    def testsets(self) -> List[Dict[str, Any]]:
        ev = self.cfg.get("eval") or {}
        raw = ev.get("testsets") or []
        if isinstance(raw, dict):
            # 允许 {small: {mode:..., ...}} 这种写法。
            out = []
            for name, item in raw.items():
                item = dict(item or {})
                item.setdefault("name", name)
                out.append(item)
            return out
        return [dict(x or {}) for x in raw]

    def scenes(self) -> List[Dict[str, Any]]:
        perf = self.cfg.get("perf") or {}
        raw = perf.get("scenes") or []
        if isinstance(raw, dict):
            out = []
            for name, item in raw.items():
                item = dict(item or {})
                item.setdefault("scene", name)
                out.append(item)
            return out
        return [dict(x or {}) for x in raw]

    def manifest_dir(self) -> Path:
        ev = self.cfg.get("eval") or {}
        p = self.path(_first_present(ev.get("manifest_dir"), "sherpa_eval/data"))
        assert p is not None
        d = Path(p)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def manifest_for_testset(self, ts: Dict[str, Any]) -> Path:
        if ts.get("manifest"):
            p = self.path(ts.get("manifest"))
            assert p is not None
            return Path(p)
        name = ts.get("name")
        if not name:
            raise SystemExit(f"eval.testsets 中有条目缺少 name: {ts}")
        return self.manifest_dir() / f"{name}.jsonl"

    # ─── manifest stage ───────────────────────────────────────────────

    def build_manifests(self, only: Optional[set] = None) -> None:
        sets = self.testsets()
        if only:
            sets = [x for x in sets if x.get("name") in only]
        if not sets:
            print("[warn] config.yaml 里没有 eval.testsets，跳过 manifest stage")
            return
        for ts in sets:
            name = ts.get("name")
            if not name:
                raise SystemExit(f"eval.testsets 中有条目缺少 name: {ts}")
            if ts.get("manifest") and not ts.get("mode"):
                print(f"[info] testset={name} 已指定 manifest，跳过构建: {ts['manifest']}")
                continue
            mode = ts.get("mode")
            if not mode:
                raise SystemExit(f"testset={name} 缺少 mode。可选: transcript / auto-pair / fixed-text，或只指定 manifest。")
            mode = str(mode).lower()
            out = self.manifest_for_testset(ts)
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd: List[str] = [
                self.python, str(ROOT / "sherpa_eval" / "build_manifest.py"),
                "--audio-dir", self.path(ts.get("audio_dir"), must=True) or "",
                "--output", str(out),
            ]
            if ts.get("ext"):
                cmd += ["--ext", str(ts["ext"])]
            if ts.get("recursive") is False:
                cmd.append("--no-recursive")
            if ts.get("upper") is False:
                cmd.append("--no-upper")
            if ts.get("space_to_underscore") is False:
                cmd.append("--no-space-to-underscore")
            if _truthy(ts.get("include_empty_text")):
                cmd.append("--include-empty-text")
            if ts.get("id_mode"):
                cmd += ["--id-mode", str(ts["id_mode"])]

            if mode == "transcript":
                cmd += ["--transcript", self.path(ts.get("transcript"), must=True) or ""]
            elif mode in ("auto-pair", "auto_pair"):
                cmd.append("--auto-pair")
            elif mode in ("fixed-text", "fixed_text"):
                cmd += ["--fixed-text", str(_first_present(ts.get("text"), ts.get("fixed_text"), ""))]
            else:
                raise SystemExit(f"testset={name} 未知 mode={mode!r}")
            self.run(cmd)

    # ─── eval / sweep stage ───────────────────────────────────────────

    def run_eval(self, *, sweep: bool = False, only: Optional[set] = None) -> None:
        sets = self.testsets()
        if only:
            sets = [x for x in sets if x.get("name") in only]
        if not sets:
            print("[warn] config.yaml 里没有 eval.testsets，跳过 eval/sweep stage")
            return
        model = self._require_model()
        common = self._common_eval_values()
        ev = self.cfg.get("eval") or {}
        if sweep:
            thresholds = _as_list(ev.get("thresholds"))
            if not thresholds:
                print("[warn] eval.thresholds 为空，跳过 sweep stage")
                return
        else:
            thresholds = [common.get("keywords_threshold")]

        for ts in sets:
            name = ts.get("name")
            if not name:
                raise SystemExit(f"eval.testsets 中有条目缺少 name: {ts}")
            manifest = self.manifest_for_testset(ts)
            for th in thresholds:
                suffix = str(common["suffix"])
                if sweep:
                    suffix = f"{suffix}-t{_stage_value(th)}"
                cmd: List[str] = [
                    self.python, str(ROOT / "sherpa_eval" / "sherpa_onnx_kws_eval.py"),
                    "--tokens", model["tokens"],
                    "--encoder", model["encoder"],
                    "--decoder", model["decoder"],
                    "--joiner", model["joiner"],
                    "--keywords-file", model["keywords_file"],
                    "--manifest", str(manifest),
                    "--testset", str(name),
                    "--suffix", suffix,
                    "--output-dir", str(self.metrics_dir),
                    "--chunk-seconds", str(common["chunk_seconds"]),
                    "--num-threads", str(common["num_threads"]),
                    "--provider", str(common["provider"]),
                ]
                if th not in (None, ""):
                    cmd += ["--keywords-threshold", str(th)]
                if common.get("keywords_score") not in (None, ""):
                    cmd += ["--keywords-score", str(common["keywords_score"])]
                if common.get("num_trailing_blanks") not in (None, ""):
                    cmd += ["--num-trailing-blanks", str(common["num_trailing_blanks"])]
                if common.get("max_active_paths") not in (None, ""):
                    cmd += ["--max-active-paths", str(common["max_active_paths"])]
                if common.get("limit") not in (None, ""):
                    cmd += ["--limit", str(common["limit"])]
                if _truthy(common.get("test_only_keywords")):
                    cmd.append("--test-only-keywords")
                if common.get("phrase_space_to_underscore") is False:
                    cmd.append("--no-phrase-space-to-underscore")
                if common.get("debug_lists") is False:
                    cmd.append("--no-debug-lists")
                cmd += _arg_list(ev.get("extra_args"))
                cmd += _arg_list(ts.get("extra_eval_args"))
                self.run(cmd)

    # ─── perf stage ───────────────────────────────────────────────────

    def _perf_manifest(self, scene: Dict[str, Any]) -> Path:
        perf = self.cfg.get("perf") or {}
        # 场景级 manifest / testset 优先。
        if scene.get("manifest"):
            p = self.path(scene.get("manifest"))
            assert p is not None
            return Path(p)
        if scene.get("testset"):
            name = str(scene["testset"])
            for ts in self.testsets():
                if ts.get("name") == name:
                    return self.manifest_for_testset(ts)
            return self.manifest_dir() / f"{name}.jsonl"
        if perf.get("manifest"):
            p = self.path(perf.get("manifest"))
            assert p is not None
            return Path(p)
        if perf.get("testset"):
            name = str(perf["testset"])
            for ts in self.testsets():
                if ts.get("name") == name:
                    return self.manifest_for_testset(ts)
            return self.manifest_dir() / f"{name}.jsonl"
        sets = self.testsets()
        if len(sets) == 1:
            return self.manifest_for_testset(sets[0])
        raise SystemExit(
            f"perf scene={scene.get('scene') or scene.get('name')} 缺少 manifest。"
            "请在 perf.manifest / perf.testset / scene.manifest / scene.testset 中指定。"
        )

    def run_perf(self, only: Optional[set] = None) -> None:
        scenes = self.scenes()
        if only:
            scenes = [x for x in scenes if (x.get("scene") or x.get("name")) in only]
        if not scenes:
            print("[warn] config.yaml 里没有 perf.scenes，跳过 perf stage")
            return
        model = self._require_model()
        ev_common = self._common_eval_values()
        perf = self.cfg.get("perf") or {}
        dec = self.cfg.get("decode") or {}

        for scene in scenes:
            scene_name = scene.get("scene") or scene.get("name")
            mode = scene.get("mode")
            if not scene_name or not mode:
                raise SystemExit(f"perf.scenes 条目必须包含 scene/name 和 mode: {scene}")
            mode = str(mode)
            manifest = self._perf_manifest(scene)
            suffix = _first_present(scene.get("suffix"), perf.get("suffix"), ev_common.get("suffix"), "onnx")
            provider = _first_present(scene.get("provider"), perf.get("provider"), ev_common.get("provider"), "cpu")
            num_threads = _first_present(scene.get("num_threads"), perf.get("num_threads"), ev_common.get("num_threads"), 1)
            chunk_seconds = _first_present(scene.get("chunk_seconds"), perf.get("chunk_seconds"), ev_common.get("chunk_seconds"), 0.5)
            threshold = _first_present(scene.get("keywords_threshold"), perf.get("keywords_threshold"), ev_common.get("keywords_threshold"), dec.get("keywords_threshold"))
            score = _first_present(scene.get("keywords_score"), perf.get("keywords_score"), ev_common.get("keywords_score"), dec.get("keywords_score"))

            cmd: List[str] = [
                self.python, str(ROOT / "sherpa_perf" / "sherpa_onnx_kws_perf.py"),
                "--tokens", model["tokens"],
                "--encoder", model["encoder"],
                "--decoder", model["decoder"],
                "--joiner", model["joiner"],
                "--keywords-file", model["keywords_file"],
                "--manifest", str(manifest),
                "--scene", str(scene_name),
                "--suffix", str(suffix),
                "--output-dir", str(self.metrics_dir),
                "--mode", mode,
                "--chunk-seconds", str(chunk_seconds),
                "--num-threads", str(num_threads),
                "--provider", str(provider),
            ]
            if threshold not in (None, ""):
                cmd += ["--keywords-threshold", str(threshold)]
            if score not in (None, ""):
                cmd += ["--keywords-score", str(score)]
            if scene.get("num_trailing_blanks") not in (None, ""):
                cmd += ["--num-trailing-blanks", str(scene["num_trailing_blanks"])]
            elif perf.get("num_trailing_blanks") not in (None, ""):
                cmd += ["--num-trailing-blanks", str(perf["num_trailing_blanks"])]
            if scene.get("max_active_paths") not in (None, ""):
                cmd += ["--max-active-paths", str(scene["max_active_paths"])]
            elif perf.get("max_active_paths") not in (None, ""):
                cmd += ["--max-active-paths", str(perf["max_active_paths"])]
            if scene.get("limit") not in (None, ""):
                cmd += ["--limit", str(scene["limit"])]
            elif perf.get("limit") not in (None, ""):
                cmd += ["--limit", str(perf["limit"])]
            if scene.get("tag") not in (None, ""):
                cmd += ["--tag", str(scene["tag"])]
            if scene.get("warmup") not in (None, ""):
                cmd += ["--warmup", str(scene["warmup"])]
            elif perf.get("warmup") not in (None, ""):
                cmd += ["--warmup", str(perf["warmup"])]

            if mode == "concurrent":
                cmd += ["--concurrency", str(_first_present(scene.get("concurrency"), 1))]
                cmd += ["--duration-seconds", str(_first_present(scene.get("duration_seconds"), perf.get("duration_seconds"), 30))]
                cmd += ["--pacing", str(_first_present(scene.get("pacing"), perf.get("pacing"), "full"))]
            elif mode == "batch":
                cmd += ["--batch-size", str(_first_present(scene.get("batch_size"), 8))]
                cmd += ["--n-batches", str(_first_present(scene.get("n_batches"), 20))]
            elif mode == "batch_streaming":
                cmd += ["--concurrency", str(_first_present(scene.get("concurrency"), scene.get("batch_size"), 8))]
                cmd += ["--duration-seconds", str(_first_present(scene.get("duration_seconds"), perf.get("duration_seconds"), 30))]
                cmd += ["--pacing", str(_first_present(scene.get("pacing"), perf.get("pacing"), "full"))]
            elif mode == "cpu_sweep":
                cmd += ["--inner-mode", str(_first_present(scene.get("inner_mode"), "concurrent"))]
                cmd += ["--concurrency-list", str(_first_present(scene.get("concurrency_list"), "1,2,4,8,16"))]
                cmd += ["--target-cpu", str(_first_present(scene.get("target_cpu"), 70.0))]
                cmd += ["--cpu-budget-mode", str(_first_present(scene.get("cpu_budget_mode"), "per_core"))]
                cmd += ["--duration-seconds", str(_first_present(scene.get("duration_seconds"), perf.get("duration_seconds"), 30))]
                cmd += ["--pacing", str(_first_present(scene.get("pacing"), perf.get("pacing"), "full"))]
                if scene.get("cpu_affinity") not in (None, ""):
                    cmd += ["--cpu-affinity", str(scene["cpu_affinity"])]
            elif mode == "single":
                pass
            else:
                raise SystemExit(f"未知 perf mode={mode!r}")

            cmd += _arg_list(perf.get("extra_args"))
            cmd += _arg_list(scene.get("extra_args"))
            self.run(cmd)

    # ─── 上层脚手架 stage ──────────────────────────────────────────────

    def parse_metrics(self) -> None:
        self.run([self.python, str(ROOT / "scripts" / "parse_decode.py"), str(self.exp_dir)])

    def register(self) -> None:
        self.run([self.python, str(ROOT / "scripts" / "update_registry.py"), str(self.exp_dir)])

    def report(self) -> None:
        self.run([self.python, str(ROOT / "scripts" / "build_report.py")])


def parse_stages(s: str) -> List[str]:
    s = s.strip()
    if s == "all":
        return list(DEFAULT_STAGE)
    if s == "full":
        return list(FULL_STAGE)
    out: List[str] = []
    for x in re.split(r"[,\s]+", s):
        x = x.strip()
        if not x:
            continue
        if x not in VALID_STAGES:
            raise SystemExit(f"未知 stage={x!r}；可选: {', '.join(FULL_STAGE)}；别名: all/full")
        out.append(x)
    return out


def resolve_config_arg(arg: str) -> Path:
    p = Path(arg)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    if p.is_dir():
        p = p / "config.yaml"
    if not p.exists():
        raise SystemExit(f"config 不存在: {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="实验 config.yaml，或实验目录 runs/expNNN_xxx")
    ap.add_argument("--stage", default="all",
                    help="逗号分隔 stage；默认 all=manifest,eval,perf,parse,register,report；full 额外包含 sweep")
    ap.add_argument("--only-testset", default="",
                    help="只处理这些 testset，逗号分隔；作用于 manifest/eval/sweep")
    ap.add_argument("--only-scene", default="",
                    help="只处理这些 perf scene，逗号分隔")
    ap.add_argument("--python", default=sys.executable or "python3", help="Python 解释器")
    ap.add_argument("--dry-run", action="store_true", help="只打印将执行的命令，不真正运行")
    args = ap.parse_args()

    config_path = resolve_config_arg(args.config)
    cfg = load_yaml(config_path)
    r = Runner(config_path, cfg, python=args.python, dry_run=args.dry_run)
    stages = parse_stages(args.stage)
    only_ts = set(_as_list(args.only_testset)) if args.only_testset else None
    only_scene = set(_as_list(args.only_scene)) if args.only_scene else None

    print(f"[info] ROOT       = {ROOT}")
    print(f"[info] config     = {_display_path(config_path)}")
    print(f"[info] EXP_DIR    = {_display_path(r.exp_dir)}")
    print(f"[info] metrics    = {_display_path(r.metrics_dir)}")
    print(f"[info] stages     = {','.join(stages)}")
    if args.dry_run:
        print("[info] dry-run    = true")

    for st in stages:
        print(f"\n========== Stage: {st} ==========")
        if st == "manifest":
            r.build_manifests(only=only_ts)
        elif st == "eval":
            r.run_eval(sweep=False, only=only_ts)
        elif st == "sweep":
            r.run_eval(sweep=True, only=only_ts)
        elif st == "perf":
            r.run_perf(only=only_scene)
        elif st == "parse":
            r.parse_metrics()
        elif st == "register":
            r.register()
        elif st == "report":
            r.report()
        else:
            raise AssertionError(st)

    print("\n[done] run_from_config 完成")


if __name__ == "__main__":
    main()
