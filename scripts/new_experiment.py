#!/usr/bin/env python3
"""
新建一次 KWS 消融实验。

用法:
    python scripts/new_experiment.py --name lr5e-5 --variable lr --value 5e-5
    python scripts/new_experiment.py --name large_ctx --variable left_context --value 128 --notes "更长上下文"

会:
  1. 在 runs/ 下分配下一个 expNNN 子目录,如 runs/exp003_lr5e-5/
  2. 拷贝 templates/config.yaml,并填好 meta 部分
  3. 建好 metrics/ 子目录
"""

from __future__ import annotations

import argparse
import datetime
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
TEMPLATE = ROOT / "templates" / "config.yaml"


def next_exp_id() -> str:
    """扫描 runs/ 下已有的 expNNN_*,返回下一个未占用的 expNNN。"""
    RUNS_DIR.mkdir(exist_ok=True)
    pattern = re.compile(r"^exp(\d{3})(?:_.*)?$")
    used = []
    for p in RUNS_DIR.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m:
            used.append(int(m.group(1)))
    n = max(used) + 1 if used else 1
    return f"exp{n:03d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="短名,如 lr5e-5,会拼到目录名里")
    ap.add_argument("--variable", default="", help="本次消融改动的变量名")
    ap.add_argument("--value", default="", help="变量取值(字符串)")
    ap.add_argument("--notes", default="", help="自由备注")
    args = ap.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f"模板不存在:{TEMPLATE}")

    exp_id = next_exp_id()
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.name).strip("_")
    exp_dir = RUNS_DIR / f"{exp_id}_{safe_name}"
    exp_dir.mkdir(parents=True)
    (exp_dir / "metrics").mkdir()

    cfg_dst = exp_dir / "config.yaml"
    shutil.copy(TEMPLATE, cfg_dst)

    # 简单替换 meta 字段,不引入 yaml 依赖
    text = cfg_dst.read_text(encoding="utf-8")
    today = datetime.date.today().isoformat()
    replacements = [
        (r'(exp_id:\s*)\S+', rf'\g<1>{exp_id}'),
        (r'(name:\s*)\S.*', rf'\g<1>{safe_name}'),
        (r'(date:\s*)"[^"]*"', rf'\g<1>"{today}"'),
        (r'(variable:\s*)\S.*', rf'\g<1>{args.variable or "_"}'),
        (r'(value:\s*)"[^"]*"', rf'\g<1>"{args.value}"'),
        (r'(notes:\s*)""', rf'\g<1>"{args.notes}"'),
    ]
    for pat, repl in replacements:
        text = re.sub(pat, repl, text, count=1)
    cfg_dst.write_text(text, encoding="utf-8")

    print(f"已创建 {exp_dir}")
    print(f"  - {cfg_dst}")
    print(f"  - {exp_dir / 'metrics'}/   <- 把 decode.py 产出的 metric-*.txt 放这里")
    print(f"\n下一步:")
    print(f"  1. 编辑 {cfg_dst} 填入真实超参")
    print(f"  2. 跑训练 + decode,把 metric-<testset>-{{pt,onnx}}[-tag].txt 放到 metrics/")
    print(f"  3. python scripts/parse_decode.py {exp_dir.relative_to(ROOT)}")
    print(f"  4. python scripts/update_registry.py {exp_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
