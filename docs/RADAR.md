# RADAR v5.1：置信度校准的相对风险软修复

RADAR v5.1 是 iTransformer 的可选前置模块。它遵循同一条处理链：**识别高依赖但低可靠的风险观测，只按有绝对证据支撑的相对风险强度修复这些块，再把修复后的原始时序交给 forecasting backbone。**

完整流程为：非等距切块 → 近期依赖先验 → 局部分布可靠性 → 联合风险 → 实例级相对风险归一化 → 原始风险置信度校准 → 分布级仿射软修复 → iTransformer。

v5.1 保留 v5.0 的实例级相对排序，但不再把 Min-Max 结果直接当作修复强度。归一化结果还要乘以原始联合风险，使“这个块在本实例中最危险”和“它确实具有足够风险证据”同时成立。这样不会因为健康序列中存在极小的风险差异，就强制制造一个接近 1 的修复块。模块仍不使用绝对阈值、Top-K、Hard Mask、均值平滑、Imputer 或辅助重建损失。每个 batch 只执行一次 forecasting backbone，RADAR 没有可训练参数，也不会读取预测标签或 decoder 的未来占位值。

## 启用与 Colab 运行

推荐使用 Python 3.12。当前 `requirements.txt` 使用兼容现代 Python 的依赖范围，并显式包含 `einops` 和 `reformer-pytorch==1.4.4`。在 Colab 中先同步项目，再安装依赖：

```python
%pip install -r /content/iTransformer_RADAR/requirements.txt
```

关闭 RADAR 的基线：

```python
%cd /content/iTransformer_RADAR
!bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh
```

启用 RADAR v5.1：

```python
%cd /content/iTransformer_RADAR
!bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh --use_radar 1
```

ETTh1 脚本依次运行预测长度为 96、192、336 和 720 的四组实验，默认使用 GPU 0 和 batch size 128。附加参数会转发到四组实验，并位于各命令末尾，因此可以覆盖脚本默认值：

```python
!bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh \
  --use_radar 1 \
  --batch_size 64 \
  --num_workers 2
```

直接调用 `run.py` 时，全局 batch size 默认仍为 32。`--use_radar` 默认值为 0，目前只支持 `--model iTransformer`。v5.1 没有绝对阈值参数；启用后的 checkpoint 和结果目录统一带 `_RADAR_v5_1` 后缀，与旧版本产物隔离。

### 末端置零扰动

可选参数 `--perturb_last_point 1` 会把每个输入窗口中所有变量的最后一个观测值置为 0。执行顺序为：

```text
原输入 → 最后一个观测点置零 → 可选 RADAR 修复 → iTransformer
```

因此，同一扰动既可用于普通 iTransformer，也可用于 RADAR 对照实验：

```python
# 扰动后的 iTransformer 基线
!bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh \
  --perturb_last_point 1

# 相同扰动先经过 RADAR，再进入 iTransformer
!bash ./scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh \
  --use_radar 1 \
  --perturb_last_point 1
```

该参数默认是 0。开启后会作用于训练、验证、测试和预测阶段的 encoder 输入，且不会原地修改 DataLoader 提供的张量。实验目录增加 `_LastPointZero` 后缀，避免覆盖无扰动实验。

## 1. 非等距切块与扩张感受野

`seq_len=96` 时，从最近到最早的块长度为：

```text
2, 2, 4, 4, 8, 16, 28, 32
```

代码按时间正序保存的真实块区间为：

```text
[0,32), [32,60), [60,76), [76,84),
[84,88), [88,92), [92,94), [94,96)
```

真实块用于风险归一化和最终修复。计算局部统计量时，长度小于 8 的末端短块使用一个以块末端为右边界、只观察过去的 8 点感受野。对应统计区间为：

```text
[0,32), [32,60), [60,76), [76,84),
[80,88), [84,92), [86,94), [88,96)
```

这样可以避免直接用 2 或 4 个样本估计局部分布。其他 `seq_len` 按 96 点模板同比例缩放并分配整数余数；所有真实块完整覆盖历史且互不重叠。

## 2. 高依赖、低可靠的联合风险

每个样本、每个变量独立得到形状为 `[B,K,C]` 的风险张量，其中 `K` 是 Chunk 数。

### 2.1 近期依赖先验

设真实块 `i` 的右边界为 `e_i`，历史长度为 `L`，块与预测起点的间隔为 `g_i=L-e_i`。依赖度定义为：

\[
D_i=\exp\left(-\lambda\frac{g_i}{L}\right), \qquad \lambda=3.
\]

最近块的 `D=1`，越早的块依赖度按指数衰减。该先验由块位置直接得到，不额外运行 backbone，也不提取 iTransformer Attention 或输入梯度。

### 2.2 局部 Anchor 与趋势感知可靠性

设块的统计感受野从 `a_i` 开始。Anchor 使用它前面紧邻的最多 32 个历史点：

\[
A_i=[\max(0,a_i-32),a_i).
\]

局部窗口和 Anchor 都计算原始均值、标准差，并通过最小二乘直线去除局部趋势，得到残差标准差 `v_i` 与 `v_i^A`。Anchor 直线从 Anchor 中心外推到局部感受野中心：

\[
\widehat\mu_i^A=\mu_i^A+b_i^A(c_i-c_i^A).
\]

这使线性趋势不会仅因块位置不同而被判为异常。最早的块没有更早上下文，其可靠性固定为 1。实现保留一维 Gaussian 2-Wasserstein 诊断量：

\[
W_{2,i}^2=(\mu_i-\widehat\mu_i^A)^2+(v_i-v_i^A)^2.
\]

可靠性打分使用无量纲的均值偏移和单向残差方差突增。设局部窗口与 Anchor 的平均原始方差为：

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

分母为 0 时对应分量定义为 0。联合风险为：

\[
\operatorname{Risk}_i=D_i(1-R_i).
\]

## 3. 实例级相对风险与绝对证据校准

先对每个 `(batch, variate)` 独立沿 Chunk 维归一化：

\[
r_{\min}=\min_i \operatorname{Risk}_i, \qquad
r_{\max}=\max_i \operatorname{Risk}_i,
\]

\[
N_i=\frac{\operatorname{Risk}_i-r_{\min}}
{r_{\max}-r_{\min}+\epsilon_r},
\qquad \epsilon_r=10^{-5}.
\]

`N` 表示实例内部的相对风险位置。最安全块严格得到 `N=0`；风险跨度明显大于 `1e-5` 时，最危险块的 `N` 非常接近 1。有限风险跨度下最大值是 `span/(span+1e-5)`。若所有块风险相同，则所有 `N` 都为 0。

v5.0 直接使用 `N` 修复，导致任意非恒定风险分布都会产生近单位强度，即使原始最大风险只有 `0.006`。v5.1 使用原始风险进行置信度校准：

\[
S_i=N_i\operatorname{Risk}_i.
\]

最终修复强度满足 `0 <= S_i <= Risk_i <= 1`。相对排名负责定位，原始风险负责判断证据强弱。健康序列即使存在相对最高块，只要原始风险很小，修复也很小；真正的高依赖低可靠块同时具有较大的 `N_i` 和 `Risk_i`。整个过程仍不执行排序、Top-K、阈值比较或布尔门控。

## 4. 分布级仿射软修复

对真实块 `C_i` 计算总体均值和总体标准差：

\[
\mu_i=\frac{1}{\ell_i}\sum_t x_{i,t},
\qquad
\sigma_i=\sqrt{\frac{1}{\ell_i}\sum_t(x_{i,t}-\mu_i)^2}.
\]

每个样本、每个变量在完整的已观测输入历史上计算全局基准：

\[
\mu_g=\frac{1}{L}\sum_{t=0}^{L-1}x_t,
\qquad
\sigma_g=\sqrt{\frac{1}{L}\sum_{t=0}^{L-1}(x_t-\mu_g)^2}.
\]

非零方差块的标准化走势为：

\[
Z_{i,t}=\frac{x_{i,t}-\mu_i}{\sigma_i}.
\]

常数块使用稳定分母且得到 `Z=0`。随后以校准后的修复强度 `S_i` 融合块统计量和全局统计量：

\[
\mu_i^{safe}=(1-S_i)\mu_i+S_i\mu_g,
\]

\[
\sigma_i^{safe}=(1-S_i)\sigma_i+S_i\sigma_g,
\]

\[
\widehat x_{i,t}=Z_{i,t}\sigma_i^{safe}+\mu_i^{safe}.
\]

代码使用与上式代数等价的修正形式：

\[
\widehat x_{i,t}=x_{i,t}+S_i
\left[Z_{i,t}(\sigma_g-\sigma_i)+(\mu_g-\mu_i)\right].
\]

该形式从原值开始，只乘一次连续强度 `S_i`。因此 `S_i=0` 时输出逐值严格保留原输入；`S_i` 接近 1 时，块的位置与尺度接近完整历史基准；中间风险块按连续强度修复。对非零方差块，变换保持块内标准化形状、点的相对次序和转折位置。

Chunk 统计通过预缓存的块归属矩阵和批量矩阵乘法完成；归一化、广播和仿射重构都是 PyTorch 张量运算。风险归一化与修复路径均保留在 autograd 图中。常数块、单点块和混合精度输入使用稳定分母与 float32 统计，输出恢复原输入 dtype。

## 默认内部设置

v5.1 只提供外部开关 `--use_radar`。内部默认值位于 `layers/RADAR.py`：

| 设置 | 默认值 |
| --- | ---: |
| 短块统计感受野 | 8 |
| 局部 Anchor 长度 | 32 |
| 近期依赖衰减 `lambda` | 3.0 |
| 可靠性温度 `gamma` | 1.0 |
| 均值差权重 | 0.01 |
| 残差方差差权重 | 1.0 |
| 风险归一化稳定项 `epsilon_r` | `1e-5` |
| 常数块标准化稳定项 | `1e-5` |
| 全局基准 | 当前完整历史窗口的总体均值/标准差 |
| 可训练参数 | 0 |
| 额外 backbone 前向/反向 | 0 |

## 诊断信息

详细诊断默认关闭，避免在 Traffic 等高维数据集上保留大量 GPU 张量。分析单个 batch 时可以显式启用：

```python
model.radar.set_diagnostics(True)
prediction = model(x, x_mark, dec_inp, y_mark)
diagnostics = model.radar.last_diagnostics

dependency = diagnostics['dependency']                    # [B, K, C]
reliability = diagnostics['reliability']                  # [B, K, C]
risk = diagnostics['risk']                                # [B, K, C]
normalized_risk = diagnostics['normalized_risk']          # [B, K, C]
repair_strength = diagnostics['repair_strength']          # [B, K, C] = normalized_risk * risk
risk_min = diagnostics['risk_min']                        # [B, 1, C]
risk_max = diagnostics['risk_max']                        # [B, 1, C]
chunk_mean = diagnostics['chunk_mean']                    # [B, K, C]
chunk_std = diagnostics['chunk_std']                      # [B, K, C]
global_mean = diagnostics['global_mean']                  # [B, 1, C]
global_std = diagnostics['global_std']                    # [B, 1, C]
safe_mean = diagnostics['safe_mean']                      # [B, K, C]
safe_std = diagnostics['safe_std']                        # [B, K, C]
standardized = diagnostics['standardized']                # [B, L, C]
point_strength = diagnostics['point_repair_strength']     # [B, L, C]
local_mean = diagnostics['local_mean']
local_std = diagnostics['local_std']
local_residual_std = diagnostics['local_residual_std']
anchor_mean = diagnostics['anchor_mean']
anchor_projected_mean = diagnostics['anchor_projected_mean']
anchor_std = diagnostics['anchor_std']
anchor_residual_std = diagnostics['anchor_residual_std']
risk_distance = diagnostics['risk_distance']
wasserstein_squared = diagnostics['wasserstein_squared']

model.radar.set_diagnostics(False)
```

诊断张量均已脱离计算图，但仍位于模型设备。DataParallel 的副本诊断不会自动汇总到主模块；需要逐样本分析时建议使用单设备或在调用处显式收集。

## 训练、checkpoint 与版本边界

RADAR v5.1 是确定性的无参数前置模块。训练损失仍是原始 forecasting loss，不包含 RADAR 重建损失。启用与关闭 RADAR 时 backbone 参数名称相同，原始 iTransformer checkpoint 可以严格加载到 v5.1 模型。

v5.1 的实验目录后缀是 `_RADAR_v5_1`。从旧 v4 命令迁移时，应删除原有的绝对阈值选项。旧 RADAR v1 checkpoint 若包含 `radar.imputer.*`，这些键在 v5.1 中没有对应组件；迁移时应丢弃旧 Imputer 键，并重新评估或训练 backbone。

输入必须是有限浮点观测。检测到 NaN 或 Inf 时模块会直接报错，避免无效值污染整块。缺失值应在数据预处理阶段按数据集定义编码。

实例级 Min-Max 仍会在每个风险不完全相同的序列中产生一个相对最高块，但原始风险校准会把健康序列的实际修复幅度压回低水平。分布变化仍不必然代表错误观测，完整历史均值也可能滞后于真实趋势，因此需要分别报告干净预测精度、受扰动鲁棒性、各 Chunk 的原始风险、相对风险和最终修复强度。

本地测试：

```bash
python -m unittest discover -s tests -v
```
