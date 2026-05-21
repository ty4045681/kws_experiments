# sherpa-onnx KWS 评测脚本

在你自己的音频数据集上,用导出的 ONNX 模型跑 KWS,产出**与 icefall
`decode.py` 完全同格式**的 `metric-*.txt`,直接对接上层实验脚手架。

## 安装

```bash
pip install sherpa-onnx numpy
```

## 一分钟上手

```bash
# 1. 把音频和转写整理成 manifest(三种模式任选一种;默认递归扫描子目录)
#    A. 已有整份 transcript
python build_manifest.py \
    --audio-dir  /data/my_wavs/ \
    --transcript /data/my_text \
    --output     data/my_test.jsonl

#    B. 每个 wav 旁边有同名 .txt
python build_manifest.py \
    --audio-dir /data/my_wavs/ --auto-pair \
    --output    data/my_test.jsonl

#    C. 一批 wav 全部说同一句话(批量正样本场景)
python build_manifest.py \
    --audio-dir  /data/wakeword_pos/ \
    --fixed-text "lights on" \
    --output     data/pos.jsonl

# 2. 跑评测
python sherpa_onnx_kws_eval.py \
    --tokens   model/tokens.txt \
    --encoder  model/encoder-...onnx \
    --decoder  model/decoder-...onnx \
    --joiner   model/joiner-...onnx \
    --keywords-file model/keywords.txt \
    --manifest data/my_test.jsonl \
    --testset small --suffix onnx \
    --output-dir ../runs/exp001_baseline/metrics/ \
    --keywords-threshold 0.35

# 3. 接回脚手架
cd ..
python scripts/parse_decode.py runs/exp001_baseline
python scripts/update_registry.py runs/exp001_baseline
python scripts/build_report.py
```

## 数据集格式(支持你自己的音频)

### A. JSONL(推荐)
```json
{"id":"a001","audio":"/abs/a001.wav","text":"TURN_ON_THE_LIGHTS","duration":2.1}
{"id":"n007","audio":"/abs/n007.wav","text":"GOOD_MORNING_SIR","duration":1.4}
```
- 正样本:`text` 中包含某个 keyword 短语(以下划线形式)
- 负样本:`text` 不包含任何 keyword,会用于计算 **FA / hour**(必须给 duration)
- 约定:text 中词间分隔用下划线,以便与 sherpa-onnx 触发结果做子串匹配;
  `build_manifest.py` 会自动帮你做这个转换

### B. TSV
```
/abs/a001.wav<TAB>TURN_ON_THE_LIGHTS<TAB>2.1
/abs/n007.wav<TAB>GOOD_MORNING_SIR<TAB>1.4
```

### C. 目录 + 转写
```bash
python sherpa_onnx_kws_eval.py ... \
  --audio-dir /data/wavs/ --transcript /data/text
```
`/data/text` 每行 `<utt_id> <text>`,utt_id 对应 wav 文件名(去后缀)。

## build_manifest.py 选项速查

| 选项 | 作用 |
|---|---|
| `--audio-dir DIR` | 音频根目录(必填) |
| `--transcript FILE` | 模式 A:整份转写文件 |
| `--auto-pair` | 模式 B:每个 wav 旁找同名 .txt |
| `--fixed-text "<text>"` | 模式 C:全部音频共用这段文本 |
| `--recursive` / `--no-recursive` | 是否递归进入子目录(默认递归) |
| `--upper` / `--no-upper` | 文本是否转大写(默认转) |
| `--space-to-underscore` / `--no-space-to-underscore` | 空格→下划线(默认开) |
| `--id-mode {stem,relpath}` | utt_id 用文件名还是相对路径(默认 relpath,避免子目录重名冲突) |
| `--include-empty-text` | 保留没有转写的音频,text 留空 |
| `--ext .wav` | 音频扩展名 |

## 关键词文件

直接用 sherpa-onnx 原生格式,每行 `<tokens> @<phrase>`:
```
L IGHT S _ON @lights on
H EAT _UP @heat up
y ǎn y uán @演员
```
脚本会:
- 把整个文件原样传给 sherpa-onnx 的 KeywordSpotter
- 同时把 `@` 后面的 phrase 抽出来,**做与 text 相同的规范化(大写 + 空格→下划线)**
  后用于 TP/FP/FN/TN 统计(口径和 [icefall decode.py](https://github.com/k2-fsa/icefall/blob/master/egs/gigaspeech/KWS/zipformer/decode.py) 一致)

> 如果你自己手工写了 manifest 且保留了空格,记得给评测脚本加
> `--no-phrase-space-to-underscore`,让两边保持一致。

## 判定语义

默认 `keyword in ref_text`(子串匹配),与 `decode.py` 默认行为一致。
加 `--test-only-keywords` 切换成严格匹配。

## 输出文件

放在 `--output-dir` 下,三个文件:

1. **`metric-{testset}-{suffix}.txt`** —— 和 decode.py 完全同格式,被上层 `parse_decode.py` 直接消费
2. **`triggers-{testset}-{suffix}.jsonl`** —— 逐条音频的 trigger 明细,debug 用
3. **`summary-{testset}-{suffix}.json`** —— 机读摘要(含 RTF、总时长、所有参数)

## 常用参数

| 参数 | 含义 |
|---|---|
| `--keywords-threshold 0.35` | 触发阈值(越高越保守) |
| `--keywords-score 1.0` | 关键词加成分 |
| `--chunk-seconds 0.5` | 流式 chunk 大小,模拟实时 |
| `--num-trailing-blanks 1` | 触发后等几帧静音再判定 |
| `--limit 50` | 只跑前 50 条,快速调通 |
| `--no-debug-lists` | metric 文件不打印 TP/FP/FN 明细列表 |
| `--no-phrase-space-to-underscore` | 关闭 keyword 自动加下划线(text 也别加) |

## 一些建议

- **正负样本比例**:KWS 评测里负样本时长决定了 FA/hour 的统计稳定性。建议至少 1~2 小时纯负样本。
- **多阈值扫描**:同一份模型,`for t in 0.20 0.25 0.30 0.35; do ... --suffix "onnx-t${t}"; done`,然后看 metric 文件就能画 ROC。
- **批量正样本数据**:用模式 C(`--fixed-text`)最方便,例如你给某个唤醒词录了一千条样本,放进一个目录(可分子目录),一行命令就能生成 manifest。
- **对齐 export 时的 chunk/left-context**:模型导出参数(`chunk_size=16, left_context_frames=128`)是固化在 ONNX 里的,本脚本无需重复指定,但要确保 keywords 文件用的是同一份 tokens.txt。
