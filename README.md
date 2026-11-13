# SREnbPI 剩余寿命预测区间修订实验

本包对应 C260082 修订稿的图 1–5、表 1–5、PIFR 权重敏感性分析、模拟的标签反馈对照和 FD001 配对差区间。Python 3.12.14、PyTorch 2.8.0+cpu。原始逐时点预测及汇总在 results 和 summary 内。

本版仅将随机模拟的独立训练–测试重复由 10 次扩展为 100 次，使用种子 701–800；原有种子 701–710 的结果保留，补充种子 711–800。模拟模型、训练设置、区间计算和评价方法不变。设备实验仍使用原来的 5 个种子 101、202、303、404、505，原设备结果及其统计协议不变。

## 直接核对数值

安装 requirements.txt 中的依赖后，在解压目录运行以下命令。仅重新汇总时，无须下载原始信号，也无须重新训练。

```bash
python -m pip install -r requirements.txt
python summarize.py results checked_summary
python figures.py results checked_figures
```

| 稿件内容 | 数据或脚本 | 统计单位 |
| --- | --- | --- |
| 表 1–2、权重敏感性、每 5 点反馈 | summary/simulation_runs.csv、simulation_summary.csv | 100 个独立训练–测试重复，种子 701–800，标准差用 n−1 为分母 |
| 表 3–4 | summary/device_summary.csv、device_macro_by_seed.csv | 先在每台设备内平均，再在设备间等权平均，最后计算 5 个种子间的均值和标准差 |
| 表 5 | summary/bearing_by_id.csv、device_metrics.csv | 每个留出轴承分别汇总 5 个种子 |
| FD001 配对差与寿命末期 | summary/fd_paired_bootstrap.csv、fd_last21.csv | 配对差按设备自举 10 000 次，种子 2901，末期为真实 RUL 0–20 的 21 个周期 |
| 图 4、图 5 | results/fd_0_101.csv.gz、bearing_4_101.csv.gz | 预先固定 FD001 编号 40、Bearing1_4 及种子 101 |
| 原始设备预测 | results/*.csv.gz | 每行是一台设备的一次预测，mid 为中心，half 为半宽；上下端点为 mid ± half |

simulation_summary.csv、device_summary.csv、bearing_by_id.csv 和 fd_last21.csv 使用两行列头。Python 可用 `pd.read_csv(path, header=[0,1])` 读取。PICP 单位是百分比，MPIW 单位为模拟响应单位、发动机运行周期或轴承分钟。MC 和 DE 分别表示 MC Dropout 与 Deep Ensemble。

## 全量重算

解压 C-MAPSS 数据集，使 train_FD001.txt、test_FD001.txt、RUL_FD001.txt 位于同一目录。默认使用本包附带的轴承特征表。以下命令从原始 FD001 和轴承特征重跑全部模型，最后重建表格与图片。

```bash
python run_all.py --cmapss CMAPSSData --output rerun
```

`--jobs 2` 可同时运行两项独立拟合。每项作业使用一个 CPU 计算线程。已存在的完整运行结果可供续算。若更改超参数，请指定新的输出目录。复算使用相同的随机数种子、依赖版本和 CPU 环境时最易逐项比对；不同数学库的浮点实现可能使边界样本的覆盖判定产生微小变化。

## 数据来源和处理

- C-MAPSS FD001 来自 Saxena 与 Goebel 发布的 NASA 涡扇发动机退化仿真数据。数据入口见 [NASA 数据目录](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) 与 [NASA PCoE 数据仓储第 6 项](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)。
- XJTU-SY 来自王彪等发布的滚动轴承全寿命试验。原始数据入口见 [发布方项目页](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets)。本次特征重算使用[原始 CSV 的镜像快照](https://huggingface.co/datasets/DavidNguyen/XJTU-SY_Bearing_Datasets/tree/274226336036cd8deb2c40b4ba323af2d11a7290)。读取工况 1 的 616 份 CSV，逐个校验压缩成员及输入尺寸，每份 32 768 行、水平与垂直两个通道。发布方论文 DOI 为 10.1109/TR.2018.2882682。
- 每个轴承的分钟数依次为 123、161、158、122、52。每分钟输入 16 个双通道时域特征和已运行分钟数。特征提取脚本同时接受原始 ZIP 或已解压的目录。RUL 标签只用于历史拟合、校准或事后评分。

```bash
python bearing_features.py XJTU-SY_Bearing_Datasets.zip bearing_features_rebuilt.csv
python run_all.py --cmapss CMAPSSData --bearing-features bearing_features_rebuilt.csv --output rerun_from_signals
```

## 复算设计与原个例的差异

1. 随机模拟使用正态噪声，三个情形的标准差依次为 1+5t、|1+5sin(2πt)| 和分段的 1 或 5。第二式取绝对值保证标准差非负，产生的方差与符号系数平方相同。响应以原单位评分，权重取 0、0.1、0.2、0.5。
2. SREnbPI 与未缩放 EnbPI 复用每次自举、每个袋外模型和同一测试中心。单序列模拟按时点自举，实际设备按整台自举。校准和还原共用同一尺度定义（总体标准差加 10⁻⁶），经验分位数采用逆经验分布。为避免极小袋外集合，若任一单位袋外模型少于 2 个则重抽整套索引。
3. 实际数据用冻结校准，不提前读测试标签。FD001 在 80 台历史设备上按设备切窗、归一化并拟合，20 台完整轨迹与官方 100 个独立测试终点评分。设备编号只在其所属文件内有意义。三层等长卷积与向上取整池化之后的形状为 7×72，按原网络行序重排为 6×84。不同设备从不共享同一个窗口。轴承用留一设备验证，并移除整条测试轨迹的归一化量和含最终失效时间的运行比例。
4. 原稿表 3 单台 SREnbPI 报告 97.44% 与 22.78 个周期。所附 Notebook 选取训练文件第 40 台发动机的 139 个满窗时点，保存输出为覆盖率 94.96%、平均半宽 18.86（折算平均全宽约 37.72）。两个记录所对应的原始拟合、设备或种子不能据此确认为同一次运行。修订稿的表 3 为按明确协议独立重算的多设备结果。新实验的缩放回归、训练样本抽取、网络实现、训练预算和随机数种子以修订稿及随文代码为准。

## 输入校验

以下 SHA-256 来自本次运行的输入字节。文本换行符的转换也会改变摘要。

| 文件 | SHA-256 |
| --- | --- |
| train_FD001.txt | 963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8 |
| test_FD001.txt | 3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851 |
| RUL_FD001.txt | a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca |
| bearing_features.csv | 2245b32b05e3d41a34f958dcb84998d627b97de332ee47b685fb78facbfc1514 |
