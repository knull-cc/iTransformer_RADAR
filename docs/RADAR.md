# RADAR v3.0：置信度门控与动量保留演化分解

RADAR v3.0 是 iTransformer 的可选前置滤波器。它在原始观测空间执行：非等距切块 → 近期依赖先验 → 局部分布可靠性 → 风险阈值门控 → 动量保留演化分解 → iTransformer。

v3.0 完整保留 v2.1 的切块、统计、风险打分和置信度门控，只替换修复内核：每个触发块由首尾点估计速度，以线性演化轨迹替代块均值作为基线，再对轨迹周围的平稳残差进行阻尼。它已完全移除梯度显著性、Attention 依赖分数、Hard Mask、Masked MLP Imputer 和重建辅助损失。启用后每个 batch 只执行一次 forecasting backbone；RADAR 本身没有可训练参数，也不会用未来标签或 decoder 的未来占位值。

## 启用与 Colab 运行

推荐使用 Python 3.12。当前 `requirements.txt` 使用兼容现代 Python 的依赖范围，并显式包含 `einops` 和 `reformer-pytorch==1.4.4`。在 Colab 中先同步当前项目，再安装依赖：

```python
%pip install -r /content/iTransformer_RADAR/requirements.txt
```

关闭 RADAR 的基线：

```python
%cd /content/iTransformer_RADAR
!bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh
```

启用 RADAR v3.0：

```python
%cd /content/iTransformer_RADAR
!bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh --use_radar 1
```

默认风险阈值为 0.75。可以从命令行调整，例如使用更严格的 0.8：

```python
!bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh \
  --use_radar 1 \
  --risk_threshold 0.8
```

ETTh1 脚本依次运行预测长度为 96、192、336 和 720 的四组实验，默认使用 GPU 0 和 batch size 128。附加参数会转发到四组实验，并位于各命令末尾，因此可以覆盖脚本默认值：

```python
!bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh \
  --use_radar 1 \
  --batch_size 64 \
  --num_workers 2
```

直接调用 `run.py` 时，全局 batch size 默认仍为 32。`--use_radar` 的默认值为 0；该开关目前只支持 `--model iTransformer`。`--risk_threshold` 的合法范围为 `[0,1]`，默认值为 0.75。启用后的 checkpoint 和结果目录带 `_RADAR_v3_rt<阈值>` 后缀，使不同阈值的消融实验彼此隔离，也避免与 v2/v2.1 产物冲突。

## 1. 非等距切块与扩张感受野

`seq_len=96` 时，从最近到最早的块长度保持为：

```text
2, 2, 4, 4, 8, 16, 28, 32
```

按时间顺序排列的真实块区间为：

```text
[0,32), [32,60), [60,76), [76,84),
[84,88), [88,92), [92,94), [94,96)
```

真实块只用于阈值门控和最终修复。计算局部统计量时，长度小于 8 的块使用一个以块末端为右边界、仅观察过去的 8 点感受野。对应统计区间为：

```text
[0,32), [32,60), [60,76), [76,84),
[80,88), [84,92), [86,94), [88,96)
```

因此最近的 2 点块不会直接用两个样本估计均值和方差，也不会查看输入窗口之外或未来的数据。其他 `seq_len` 按 96 点模板同比例缩放并分配整数余数；所有真实块完整覆盖历史、互不重叠。

## 2. 轻量级联合风险分数

每个样本、每个变量独立计算 `[B,K,C]` 的分数，`K` 为块数。

### 2.1 近期依赖先验

设真实块 `i` 的右边界为 `e_i`，历史长度为 `L`。块与预测起点之间的间隔为 `g_i=L-e_i`，依赖度定义为：

\[
D_i=\exp\left(-\lambda\frac{g_i}{L}\right), \qquad \lambda=3.
\]

最近块的 `D=1`，越早的块依赖度按指数衰减。该分数只由固定块位置决定，与模型架构、预测输出和输入数值无关，因此不再执行额外 backbone 前向、反向传播或 Attention 提取。

### 2.2 局部平滑 Anchor 与趋势感知可靠性

设块的统计感受野从 `a_i` 开始。Anchor 使用其前面紧邻的最多 32 个历史点：

\[
A_i=[\max(0,a_i-32),a_i).
\]

Anchor 均值就是该局部历史窗口的移动平均，不再使用全局历史均值。32 点窗口比末端短块的 8 点统计感受野更长，可降低平滑周期曲率随相位变化造成的误判。最早的块没有更早上下文时，可靠性直接设为 1，不据此修复。

局部窗口和 Anchor 都计算原始均值与标准差。为了重点识别突发高频噪声，同时避免把线性物理趋势误判为异常，两个窗口各自用最小二乘直线去除局部趋势，得到残差标准差 `v_i` 和 `v_i^A`。Anchor 直线还会从 Anchor 中心 `c_i^A` 外推到局部感受野中心 `c_i`：

\[
\widehat\mu_i^A=\mu_i^A+b_i^A(c_i-c_i^A),
\]

其中 `b_i^A` 是 Anchor 的局部斜率。这样，任意幅度和斜率的线性趋势都有 `\mu_i=\widehat\mu_i^A`，不会因块位置不同而被误判。原始的一维 Gaussian 2-Wasserstein 距离作为诊断量保留：

\[
W_{2,i}^2=(\mu_i-\widehat\mu_i^A)^2+(v_i-v_i^A)^2.
\]

可靠性打分进一步做无量纲化，并只把残差方差的**上升**视为高频噪声爆发。设 `q_i` 为局部窗口与 Anchor 的平均原始方差：

\[
q_i=\tfrac12(s_i^2+(s_i^A)^2),
\]

\[
d_{\mu,i}=\frac{(\mu_i-\widehat\mu_i^A)^2}
{(\mu_i-\widehat\mu_i^A)^2+q_i},
\qquad
d_{\sigma,i}=\frac{[\max(v_i-v_i^A,0)]^2}{q_i},
\]

\[
\widetilde W_i=0.01d_{\mu,i}+d_{\sigma,i},
\qquad R_i=\exp(-\gamma\widetilde W_i), \qquad \gamma=1.
\]

分母为 0 时对应分量定义为 0。计算不再加入带观测单位的绝对 `epsilon`，数值清零阈值也只相对于窗口方差和浮点精度，因此平移或改变计量单位不会改变可靠性。使用单向残差尺度差可避免把一个更平稳的新窗口误判为噪声，同时显著降低对干净周期信号的平滑。

联合风险为：

\[
\operatorname{Risk}_i=D_i(1-R_i).
\]

v3.0 保留 v2.1 的门控方式，不排序或限制修复块数量。对每个样本、每个变量、每个块独立应用严格阈值门控：

\[
G_i=\mathbb{1}[\operatorname{Risk}_i>\tau], \qquad \tau=0.75.
\]

只有 `Risk > risk_threshold` 的块会进入修复；风险恰好等于阈值时不触发。如果所有块都不超过阈值，则点级修复掩码全为 `False`，输出通过 `torch.where` 逐值保留原输入，形成严格的 Identity Mapping。门控是并行布尔张量比较，不执行排序、`topk` 或逐块 Python 循环。

## 3. Velocity-Preserving Evolution Decomposition

设通过门控的真实块为

\[
C_i=(x_{i,0},x_{i,1},\ldots,x_{i,\ell_i-1}),
\]

其中 $\ell_i$ 是真实块长度。首先用首尾点差分估计整个块的一阶速度：

\[
V_i=\frac{x_{i,\ell_i-1}-x_{i,0}}{\ell_i-1}.
\]

然后为块内每个位置 `t` 构建线性演化基线：

\[
T_{i,t}=x_{i,0}+tV_i, \qquad t=0,1,\ldots,\ell_i-1.
\]

平稳残差及其风险自适应阻尼为：

\[
E_{i,t}=x_{i,t}-T_{i,t},
\qquad
\alpha_i=1-\operatorname{Risk}_i,
\qquad
\widehat E_{i,t}=\alpha_iE_{i,t}.
\]

最终重构为：

\[
\widehat x_{i,t}=T_{i,t}+\widehat E_{i,t}.
\]

由于首尾点相对于该演化基线的残差严格为 0，修复后首尾点保持不变：

\[
\widehat x_{i,0}=x_{i,0}, \qquad
\widehat x_{i,\ell_i-1}=x_{i,\ell_i-1}.
\]

因此，由首尾差定义的块级一阶动量也严格保持：

\[
\frac{\widehat x_{i,\ell_i-1}-\widehat x_{i,0}}{\ell_i-1}=V_i.
\]

未通过门控的块，其 `alpha` 强制为 1，并由最终布尔掩码逐值返回原输入。实现预先缓存块起点、终点、跨度及每个点的块内偏移，前向阶段使用 `index_select`、广播、布尔比较和 `torch.where` 完成整批计算，不执行逐块 Python 循环。

长度为 1 的块定义 `V=0` 并安全地保持原值。长度为 2 的块只有首尾点，其线性演化基线与原块完全相同，因此残差恒为 0；按照上述公式，即使触发门控也不会修改这类块。这是首尾速度约束下没有内部自由度的直接结果。

## 默认内部设置

v3.0 提供 `--use_radar` 开关和 `--risk_threshold` 门控阈值。其余内部默认值仍位于 `layers/RADAR.py`：

| 设置 | 默认值 |
| --- | ---: |
| 风险门控阈值 `tau` | 0.75 |
| 短块统计感受野 | 8 |
| 局部 Anchor 长度 | 32 |
| 近期依赖衰减 `lambda` | 3.0 |
| 可靠性温度 `gamma` | 1.0 |
| 均值差权重 | 0.01 |
| 残差方差差权重 | 1.0 |
| 残差阻尼 | `alpha = 1 - Risk` |
| 可训练参数 | 0 |
| 额外 backbone 前向/反向 | 0 |

## 诊断信息

详细诊断默认关闭，避免在 Traffic 等高维数据集上保留大量 GPU 张量。需要分析单个 batch 时先显式启用：

```python
model.radar.set_diagnostics(True)
prediction = model(x, x_mark, dec_inp, y_mark)
diagnostics = model.radar.last_diagnostics

dependency = diagnostics['dependency']                 # [B, K, C]
reliability = diagnostics['reliability']               # [B, K, C]
risk = diagnostics['risk']                             # [B, K, C]
alpha = diagnostics['alpha']                           # [B, K, C]
velocity = diagnostics['velocity']                     # [B, K, C]
trend = diagnostics['trend']                           # [B, L, C]
selected_blocks = diagnostics['selected_blocks']       # [B, K, C], bool，Risk > tau
repair_mask = diagnostics['repair_mask']               # [B, L, C], bool
local_mean = diagnostics['local_mean']                 # 感受野统计
local_std = diagnostics['local_std']
local_residual_std = diagnostics['local_residual_std']
anchor_mean = diagnostics['anchor_mean']
anchor_projected_mean = diagnostics['anchor_projected_mean']
anchor_std = diagnostics['anchor_std']
anchor_residual_std = diagnostics['anchor_residual_std']
risk_distance = diagnostics['risk_distance']
wasserstein_squared = diagnostics['wasserstein_squared']

model.radar.set_diagnostics(False)  # 同时释放 last_diagnostics 引用
```

诊断张量均已脱离计算图，但仍位于模型设备。DataParallel 的副本诊断不会自动汇总到主模块；需要逐样本分析时建议使用单设备或在调用处显式收集。

## 训练、checkpoint 与边界

RADAR v3.0 是确定性的无参数滤波器。训练损失保持为原始 forecasting loss，不包含 RADAR 重建损失。启用与关闭 RADAR 的 backbone 参数名称相同，原始 iTransformer checkpoint 可以严格加载到 v3.0 模型。

旧 RADAR v1 checkpoint 如果包含 `radar.imputer.*`，这些键在 v3.0 中已不存在。v3.0 使用新的 `_RADAR_v3_rt<阈值>` 实验目录，不会覆盖旧版结果。手动迁移 v1 checkpoint 时应丢弃这些旧键并重新评估/训练 backbone；v1 的 Imputer 权重没有对应的 v3.0 组件。

输入必须是有限浮点观测；RADAR 在检测到 NaN/Inf 时会直接报错，避免一个无效值污染整个块。缺失值应在数据预处理阶段按数据集定义编码。分布变化不必然代表错误观测；新的真实高频状态也可能被阻尼。`lambda`、Anchor 长度、均值权重和风险阈值仍需要在目标数据的干净场景与受扰动场景上做消融实验。

本地测试：

```bash
python -m unittest discover -s tests -v
```
