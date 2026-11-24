# Polyp-PVT 课程复现

本仓库复现 **Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers** 的核心网络，提供训练、预测、评价代码及五个测试集的实验结果。

原版模型使用 PVTv2-B2 编码器、CFM、原 CIM、Similarity Aggregation Module 和两个预测头。运行时使用 `--variant baseline`；其他变体属于扩展实验，本页结果均来自原版 baseline。

## 实验结果

原版模型完成100轮训练，以内部验证集 soft Dice 选择第94轮权重，最佳验证 soft Dice 为 **0.9462947**。

| 测试集 | 图像数 | meanDic | meanIoU |
|---|---:|---:|---:|
| CVC-300 | 60 | 0.8704 | 0.8071 |
| CVC-ClinicDB | 62 | 0.9305 | 0.8827 |
| Kvasir | 100 | 0.9150 | 0.8619 |
| CVC-ColonDB | 380 | 0.8040 | 0.7255 |
| ETIS-LaribPolypDB | 196 | 0.7393 | 0.6676 |
| 五集等权平均 | 798（总数） | **0.8518** | **0.7890** |

五集均值按数据集等权计算，不按图像数加权。验证集 soft Dice 与测试 meanDic 采用不同计算口径，分别报告。

结果文件：

- [五集完整指标](results/summary.csv)
- [原论文结果对照与差值](results/paper_comparison.csv)
- [运行配置与环境摘要](results/reproduction_summary.json)

完整训练日志、逐图评价、10组真实预测样例和实际运行截图随课程报告提交。数据集、预训练权重及训练权重未包含在本仓库中。

## 代码结构

```text
README.md
requirements.txt
lib/
  pvt.py
  pvtv2.py
polyp_pvt_md_v1/
  model.py
  run.py
  protocol.py
  prepare_data.py
  md_block.py
  reference.json
  first_round.py
  evaluation/
results/
代码核查.json
```

`model.py` 在 baseline 分支直接构建 `lib.pvt.PolypPVT`。`md_block.py` 为扩展分支提供模块，现有导入结构需要保留该文件。`first_round.py` 调度多个变体；复现本页原版结果时，使用下述 `run.py --variant baseline` 命令即可。

课程代码的10份记录源码均与实际训练运行保存的 SHA256 一致，包括模型、训练器、骨干及评估器。该检查用于关联代码与运行记录，不构成对上游最新版本的独立核验。

## 实验环境

| 组件 | 实际运行版本 |
|---|---|
| Python | 3.10.12 |
| PyTorch | 2.2.1+cu121 |
| torchvision | 0.17.1+cu121 |
| timm | 0.6.13 |
| NumPy | 1.26.4 |
| SciPy | 1.15.1 |
| Pillow | 11.1.0 |

训练日志记录设备为 CUDA。安装时先配置适合本机的 PyTorch 与 CUDA 运行环境，再按 `requirements.txt` 安装其余依赖；已有匹配环境可直接使用。

## 数据与预训练权重

参考[原作者项目](https://github.com/DengPingFan/Polyp-PVT)获取数据与 PVTv2-B2 预训练权重，按以下结构放置：

```text
pretrained_pth/pvt_v2_b2.pth
dataset/TrainDataset/images/
dataset/TrainDataset/masks/
dataset/TestDataset/<数据集名>/images/
dataset/TestDataset/<数据集名>/masks/
```

五个测试目录名与结果表一致。图像与对应标注使用相同文件名主体。程序默认检查训练池1450张及五个测试集的预期样本数。

本次运行记录的 PVTv2-B2 预训练文件 SHA256 为：

```text
80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d
```

已有 `runs/polyp_pvt_md_v1/data/report.json` 和冻结清单时，复用该划分；数据准备命令用于首次创建清单。

## 运行方法

以下命令均在仓库根目录执行，`python` 指向已配置的运行环境。

### 数据准备与流程检查

```bash
python polyp_pvt_md_v1/prepare_data.py
python polyp_pvt_md_v1/run.py check --variant baseline
python polyp_pvt_md_v1/run.py smoke --variant baseline
```

准备程序将1450张训练池图像划为1305张训练与145张内部验证，并保存数据检查报告及预览。已有清单时跳过首次准备步骤。

`check` 构建 baseline、spatial 和 multidomain，核对公共初始化；`smoke` 对一批真实样本执行三尺度更新并检查两张验证图，用于验证流程。

### 训练与续训

```bash
python -u polyp_pvt_md_v1/run.py train --variant baseline
```

默认训练100轮，结果保存至：

```text
runs/polyp_pvt_md_v1/baseline_seed42/
```

中断后从检查点续训：

```bash
python -u polyp_pvt_md_v1/run.py train --variant baseline --resume
```

训练器会核对续训的数据、源码、环境及配置指纹。`--stop-after 1` 可用于在完成一个额外epoch后保存检查点并暂停；正式测试要求完成计划的100轮训练。

### 预测与评价

```bash
python -u polyp_pvt_md_v1/run.py predict --variant baseline

python -u polyp_pvt_md_v1/evaluation/evaluate.py \
  --project-root . \
  --pred-root runs/polyp_pvt_md_v1/baseline_seed42/predictions \
  --gt-root dataset/TestDataset \
  --output-dir runs/polyp_pvt_md_v1/baseline_seed42/evaluation \
  --workers 2
```

测试数据使用其他路径时，将 `--gt-root` 改为数据报告中的实际目录。预测和评价程序会检查已有输出，保留既有结果，避免覆盖。

## 训练与评价协议

- 数据：1305张训练、145张内部验证、798张测试；未提供患者或视频ID，因此仅描述为图像级划分。
- 优化：seed42、batch16、AdamW，学习率与 weight decay 均为1e-4；恒定学习率、100轮，无额外增强。
- 输入：基准尺寸352，多尺度倍率0.75/1/1.25按32对齐，实际为256/352/448；每批每尺度更新一次。
- 梯度：逐元素截断至 `[-0.5, 0.5]`。
- 损失：保留旧 `reduce='none'` 参数的实际计算行为，每个预测头采用全局平均 BCE 加 weighted IoU，再将两个预测头损失相加。
- 选模：内部验证集原分辨率 soft Dice 最大值对应的权重。
- 预测：两个预测头 logits 相加，缩放至原分辨率后应用 sigmoid、逐图 min-max 归一化，再量化为 uint8。
- 评价：使用 UACANet Python 指标实现，按256个阈值计算 meanDic、meanIoU 等指标。

## 与原论文的协议差异

本次实现从1450张训练池另留145张内部验证，并按内部验证结果选模；Python评价器与原论文 MATLAB 评价尚未验证逐项等价。因此，`results/paper_comparison.csv` 中的差值用于描述复现差异，不作为严格同条件的方法优劣比较。

本次仅报告 seed42，五集等权平均 meanDic 比论文报告值低约1.7978个百分点，meanIoU 低约1.4829个百分点；其中 ETIS 的 Dice 差距最大，约4.7691个百分点。数据划分、选模和评价实现均可能影响结果，现有单次运行不足以独立确定各因素的贡献。

## 来源与授权

- 原模型及骨干：[DengPingFan/Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT)。
- 评价指标实现：[plemeri/UACANet](https://github.com/plemeri/UACANet)。对应许可文本保留于 [LICENSE-UACANet.txt](polyp_pvt_md_v1/evaluation/LICENSE-UACANet.txt)。

第三方代码的使用与分发遵循对应上游项目的授权声明；本仓库未为原网络另行指定许可证。
