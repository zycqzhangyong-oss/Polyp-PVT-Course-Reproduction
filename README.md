# Polyp-PVT 课程复现代码

本目录从本次提供的服务器代码包中整理，保留原有模型和训练器，不修改其计算逻辑。课程复现请选择 `--variant baseline`：`model.py` 中该分支直接实例化 `lib.pvt.PolypPVT`。其他分支替换了 CIM，属于扩展实验。

## 当前验证范围

已检查服务器包的84个 Python 文件，语法均通过；两个核心网络文件的 AST 与包内 `reference.json` 一致。现已取得原版baseline真实运行结果，课程包10个记录源码文件的SHA256均与此次运行fingerprints一致。该检查不是此次重新访问官方仓库的独立上游版本核验。

服务器导出的 `baseline_65imnwpy.zip` 已核验：原版baseline完成100轮，最佳epoch94，五集共798张，等权meanDic=0.8518218566、meanIoU=0.7889707799。当前包的 `results/` 附正式汇总CSV与论文对照；相邻附件 `PolypPVT_真实复现结果/` 含日志、配置、逐图结果和10组真实预测样例。本机未独立重跑训练。数据、预训练及训练权重仍在服务器，未放入本代码包。

## 环境与准备

历史服务器环境：Python 3.10.12，PyTorch 2.2.1+cu121，torchvision 0.17.1+cu121，timm 0.6.13，numpy 1.26.4，scipy 1.15.1，Pillow 11.1.0。`requirements.txt` 固定相应包版本；新机器需根据平台另行选择合适的 PyTorch 构建。已有服务器环境无需重新安装或升级。

在项目根目录运行以下命令。将数据和 PVTv2-B2 预训练权重放置为：

```text
lib/pvt.py
lib/pvtv2.py
polyp_pvt_md_v1/
pretrained_pth/pvt_v2_b2.pth
dataset/TrainDataset/images/
dataset/TrainDataset/masks/
dataset/TestDataset/<数据集名>/images/
dataset/TestDataset/<数据集名>/masks/
```

五个测试集名称是 `CVC-300`、`CVC-ClinicDB`、`Kvasir`、`CVC-ColonDB`、`ETIS-LaribPolypDB`。程序默认要求1450张训练池图像，测试集分别60、62、100、380、196张。配对图像与标注使用相同文件名主体。

原方法及数据/权重获取说明参考原作者项目：<https://github.com/DengPingFan/Polyp-PVT>。本包不附数据或权重，不以随机权重替代预训练权重。历史记录的预训练 SHA256 为 `80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d`，需对实际文件核验。

## 已有服务器项目

如果已有 `runs/polyp_pvt_md_v1/data/report.json` 与冻结清单，先导出现有结果确认基线状态，复用原清单。不要重复准备数据，不必重训已经完成且协议相同的基线。不要将本精简目录覆盖到正在运行的项目中。

## 在独立项目中从头运行

以下仅是待执行命令，没有在本次 Mac 会话运行。`python` 应指向上述已配置好的环境。准备数据默认从1450张中留出10%内部验证集；首次生成后人工检查预览及报告。输出已存在时程序会拒绝覆盖。

```bash
python polyp_pvt_md_v1/prepare_data.py
python polyp_pvt_md_v1/run.py check --variant baseline
python polyp_pvt_md_v1/run.py smoke --variant baseline
```

`check` 会构建 baseline、spatial、multidomain 三个模型核对公共初始化，不会训练；`smoke` 只做一批真实样本的三尺度训练更新和两张验证图检查，其临时权重不作为正式结果。

确认检查通过且 GPU 可用后，正式训练：

```bash
python -u polyp_pvt_md_v1/run.py train --variant baseline
```

这会运行100轮。若只做一次流程检查，可在**新的运行目录**先使用：

```bash
python -u polyp_pvt_md_v1/run.py train --variant baseline --stop-after 1
python -u polyp_pvt_md_v1/run.py train --variant baseline --resume
```

第一条只完成计划100轮中的1轮，不能把它写成已完成正式训练。现有训练器仅允许计划全部完成后运行正式测试，因此本包没有把少轮流程检查冒充完整测试结果。

训练结束后生成五集预测：

```bash
python -u polyp_pvt_md_v1/run.py predict --variant baseline
```

然后评估（若实际测试目录不同，将 `--gt-root` 改为数据报告中的真实测试根目录）：

```bash
python -u polyp_pvt_md_v1/evaluation/evaluate.py \
  --project-root . \
  --pred-root runs/polyp_pvt_md_v1/baseline_seed42/predictions \
  --gt-root dataset/TestDataset \
  --output-dir runs/polyp_pvt_md_v1/baseline_seed42/evaluation \
  --workers 2
```

预测和评估输出目录若已存在，应先检查结果，不要删除后盲目重跑。训练器也会检查续训时的数据、源码、环境和配置指纹。

## 训练与评价协议

- 原网络结构：PVTv2-B2、CFM、原 CIM、Similarity Aggregation Module 和两个预测头。
- 训练池1450张，通常分为1305训练与145内部验证；没有患者/视频ID时仅称图像级划分。
- seed42、batch16、AdamW、学习率1e-4、weight decay 1e-4、恒定学习率、100轮，无额外增强。
- 输入352，倍率0.75/1/1.25并按32对齐，实际尺寸256/352/448；每批每尺度更新一次。
- 梯度裁剪为逐元素截断到[-0.5,0.5]，不是梯度范数裁剪。
- 损失保留旧代码 `reduce='none'` 的实际行为：全局平均BCE加weighted IoU，两个预测头相加。不能描述成已经修正的逐像素weighted BCE。
- 以内部验证原分辨率soft Dice选模；测试先输出uint8预测，再用UACANet Python评测器计算256阈值的meanDic/meanIoU等指标。
- 当前Python评估器尚未证明与论文MATLAB评估逐项等价；与论文比较时披露训练池留出验证、选模和评测器差异。

## 提交时应附

GitHub仓库链接、本 README、代码、结果表、真实运行截图、输入/真值/预测示例，以及简短的论文结果对照分析。服务器训练结果位于 `runs/polyp_pvt_md_v1/baseline_seed42/`。正式结果已导出，汇总表与论文对照在本包 `results/`，完整证据在课程附件中。GitHub仓库尚未上传，需补仓库链接。

## 来源与权利说明

原网络来源：<https://github.com/DengPingFan/Polyp-PVT>；评估器来源：<https://github.com/plemeri/UACANet>，保留 `evaluation/LICENSE-UACANet.txt`。`md_block.py` 用于扩展分支，baseline不使用该模块，但现有模块导入需要保留文件。

本次服务器导出包未包含原项目根目录LICENSE文件，因此未给原网络擅自指定许可证。公开GitHub前需一并核对上游授权并保留相应声明。此文件夹仅完成本地整理，尚未上传GitHub。

## 保留辅助文件的原因

`first_round.py` 与原运行源码哈希一致，为保持源文件清单完整而保留；该脚本会调度多个变体。课程原版复现使用上述明确的 `run.py --variant baseline` 命令，不需要执行first_round.py。
