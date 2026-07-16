# 数据目录

## RSOD：流程验证数据

RSOD 官方下载由 `aircraft`、`oiltank`、`overpass`、`playground` 四个目录组成。
使用以下命令合并图像、转换 COCO 标注，并生成固定的 70/15/15 划分：

```bash
python scripts/prepareRsod.py \
  --source "/path/to/RSOD数据集" \
  --output data/RSOD
```

脚本会保留无标注负样本、裁剪越界框，并确保完全相同的图像不会跨越
train/val/test。所有修正、重复项和划分统计均写入 `data/RSOD/manifest.json`。
原始数据不会被修改。

合并后或上传服务器后运行独立校验：

```bash
specloc validate rsod
```

RSOD 仅用于小规模流程验证。上游仓库没有明确附带数据集许可文件，因此在取得明确
授权前，不要上传或重新分发 RSOD 图像、原始标注、转换后的 COCO 标注或数据压缩包：
<https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset->。

## AI-TOD-v2：正式实验数据

请通过合法、授权的渠道获取 AI-TOD-v2，并按以下结构放置。AI-TOD 官方仓库将数据集
授权为 CC BY-NC-SA 4.0；AI-TOD-v2 使用相同图像，因此数据不得被误标为本项目的
Apache-2.0 代码。公开再分发时必须满足署名、非商业和相同方式共享等要求：
<https://github.com/jwwangchn/AI-TOD>。

```text
data/AI-TOD/
├── train/
├── val/
└── annotations/
    ├── aitod_train_v2.json
    └── aitod_val_v2.json
```

正式运行前必须生成数据 manifest：

```bash
specloc validate aitod \
  --source "<实际授权下载地址或来源说明>" \
  --license-note "<研究者已核实的许可或访问条款>"
```

校验器检查：

- train 标注包含 11,214 张图像；
- val 标注包含 5,607 张图像；
- 类别集合与官方 8 类一致；
- 标注中的全部图像路径存在；
- train/val 标注文件的 SHA256；
- 数据来源与访问条款。

成功后会生成 `data/AI-TOD/manifest.json`，其 `status` 必须为 `verified`。
数据和 manifest 均不会被 Git 跟踪；请在每台实际运行机器上重新校验。

官方项目入口：https://github.com/Chasel-Tsui/mmdet-aitod
