# RADAR 输入修复插件

RADAR 在 iTransformer 的归一化与 embedding **之前**执行：历史时序分块 → 识别高依赖、低可靠块 → 在原始观测空间补全 → iTransformer。修复结果保持 `[batch, seq_len, channels]`，历史顺序、时间标记和预测输出接口保持原样。

## 启用

在原有命令末尾添加 `--use_radar 1` 即可；也支持简写 `--use_radar`。默认 `--use_radar 0`，关闭时不创建补全器，原有 backbone 参数名称、初始化顺序和预测计算保持一致。此开关目前仅支持 `--model iTransformer`。

```bash
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des Exp \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --use_radar 1
```

训练、验证、测试和 `--do_predict` 都会自动应用输入修复。RADAR 实验的 checkpoint 和结果目录添加 `_RADAR` 后缀，避免与同配置的原始 iTransformer 实验混用。测试时使用相同开关和配置，修改 `--is_training 0`。

## 1. 统一的非等距切块

使用统一的、近期更细的几何切块模板。`seq_len=96` 时，**从最近往最早**的块长度为：

```text
2, 2, 4, 4, 8, 16, 28, 32
```

按时间顺序的区间是 `[0,32), [32,60), [60,76), [76,84), [84,88), [88,92), [92,94), [94,96)`。其他历史长度按此模板同比例缩放并分配整数余数；不足 8 步时使用单点块。所有块长度为正、完整覆盖历史、无重叠，同一历史长度的所有样本和变量共享边界。

## 2. 依赖、可靠性与风险

每个样本、每个变量独立评分和选择修复块；诊断分数形状为 `[B, K, C]`，块索引从最早到最近排列。

**依赖 `D`。** iTransformer 的 attention 表示变量间关联，不能直接作为时间块依赖。因此对未修复输入做一次 backbone 前向，并用两次固定随机符号向量的向量–雅可比积估计输入敏感度：

\[
g_t = \sqrt{\frac{1}{M}\sum_{m=1}^{M}
\left(\frac{\partial\langle v_m, f(X)\rangle}{\partial X_t}\right)^2},
\qquad D_i = \frac{\operatorname{Mean}_{t\in i} g_t}
{\sum_j \operatorname{Mean}_{t\in j}g_t+\epsilon}.
\]

`M=2`，`v_m` 是固定种子的 Rademacher 符号投影。该分数是输出雅可比范数的低成本估计，不是完整雅可比的精确计算。块内取均值，避免较长的历史块仅因长度更大获得更高分。`features=MS` 时只对最终目标变量的预测做投影。探测时临时关闭 dropout，随后恢复各模块原来的 train/eval 状态；只计算输入梯度，不向 backbone 的 `.grad` 累积梯度。

探测包含归一化均值随输入变化的梯度，避免直接沿用训练时 `mean.detach()` 而遗漏这部分敏感度。原来的原地归一化除法改为等价的非原地计算，使输入梯度探测和补全器训练可以反向传播；正常预测数值及 backbone 的参数训练行为保持一致。

**可靠性 `R`。** 使用当前输入的**较早一半**作为历史 anchor，分别计算每个变量的均值和总体标准差。按一维高斯假设计算并标准化 Wasserstein 距离：

\[
\widetilde W_{2,i}^{2}=\frac{(\mu_i-\mu_a)^2+(\sigma_i-\sigma_a)^2}
{\sigma_a^2+\epsilon},\qquad R_i=\exp(-\gamma\widetilde W_{2,i}^{2}).
\]

标准化用于减轻不同变量计量单位的影响。默认 `gamma=1`，`R_i < 0.2` 视为低可靠。方差使用 `unbiased=False` 并添加数值稳定项，因此单点块和常数序列也可以计算。

**重建误差 `E`。** 先屏蔽全部低可靠块，让补全器只使用高可靠历史。将该上下文补全与原观测的差异按 anchor 标准差归一化，并在块内计算均方误差：

\[
E_i=\operatorname{Mean}_{t\in i}\left(\frac{X_t-\widehat X_t}{\sigma_a}\right)^2,
\qquad \operatorname{Risk}_i=D_i E_i.
\]

这里遵循 `Dependency × Reconstruction Error`：Wasserstein 可靠性决定候选资格，重建误差用于联合风险排序，两者分工明确。

**选择。** 候选块同时满足低可靠 `R_i < 0.2`、依赖不低于块平均水平 `D_i >= 1/K`、风险大于零，然后按风险选取**至多 3 块**。候选不足 3 块时不凑数，没有候选时保留原输入。如果某变量的全部块都低可靠，缺少可信补全上下文，则保留该变量输入。

## 3. Data-space 补全与训练

补全器是所有变量共享的两层线性映射 MLP，中间使用 GELU，隐藏维度 64。输入包含屏蔽后的标准化历史值和独立的可见性标记；使用可见历史计算均值和标准差，隐藏值不会通过归一化泄漏。

输出是“上下文插值 + 学习残差”，随后恢复到输入的观测尺度。残差输出层初始化为零，未训练时已有确定的插值基础：内部缺口使用两侧历史插值，窗口末尾缺口延用最近可信历史值。仅选中的风险块被替换，其他有限观测逐值保留。`seq_len=96` 时补全器增加 **18,592 个参数**，不随变量数量增加。

训练时，额外随机屏蔽 1–3 个高可靠历史块，至少保留一个高可靠块作为上下文。以被屏蔽的历史观测作为自监督目标，按块平均计算标准化重建 MSE：

\[
\mathcal L=\mathcal L_{forecast}+0.1\mathcal L_{masked\ reconstruction}.
\]

补全器还可通过最终预测损失学习。低可靠原观测不作为重建标签。风险排序和硬掩码不反向传播；补全与最终预测可反向传播。验证和测试只计算预测指标，没有未来标签参与风险评分。

常规训练与 `partial_train` 两条训练路径均添加重建损失，包含原有的 attention 输出、混合精度和分变量训练分支。额外损失随前向返回并在训练端汇总，避免依赖 DataParallel 副本的临时属性。内部依赖探测可以在外层 `torch.no_grad()` / `torch.inference_mode()` 中运行。

## 默认设置和诊断

只新增启用开关；原有参数及其默认值保持不变。方法内部默认值在 `layers/RADAR.py` 的 `RADAR.__init__` 中：

| 设置 | 默认值 |
| --- | --- |
| 每个样本、每个变量最多修复的块数 | 3 |
| 低可靠阈值 | 0.2 |
| Wasserstein 衰减系数 | 1.0 |
| MLP 隐藏维度 | 64 |
| 依赖梯度投影次数 | 2 |
| 自监督重建损失权重 | 0.1 |

```python
prediction = model(x, x_mark, dec_inp, y_mark)
diagnostics = model.radar.last_diagnostics
dependency = diagnostics['dependency']
reliability = diagnostics['reliability']
reconstruction_error = diagnostics['reconstruction_error']
risk = diagnostics['risk']
selected_blocks = diagnostics['selected_blocks']  # [B, K, C], bool
repair_mask = diagnostics['repair_mask']          # [B, L, C], bool
```

这些诊断均已脱离计算图。DataParallel 时可通过 `model.module.radar` 访问基础模块，但副本诊断不会自动聚合到基础模块；需要聚合诊断时应单设备运行或自行收集。

如果使用自己的训练循环，可用 `prediction, radar_loss = model(..., return_radar_loss=True)` 取得已经乘以 `0.1` 的重建损失，再加入预测损失。普通 `forward` 的返回值保持原样，包括 `output_attention=True` 时的 `(prediction, attentions)`。

## 已有 backbone 权重

插件没有改动原始 backbone 的参数名称和结构。可以向启用 RADAR 的模型加载原始 iTransformer 权重，允许缺少新补全器的 `radar.*` 参数；这些参数仍需训练。默认实验流程联合训练 backbone 与补全器。如果希望保留预训练 backbone 的权重，可以在自己的训练入口冻结它们：

```python
missing, unexpected = model.load_state_dict(backbone_state_dict, strict=False)
assert all(name.startswith('radar.') for name in missing)
assert not unexpected
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith('radar.'))
```

冻结参数后，输入梯度依赖探测仍然可用。

## 边界与参考

当前输入要求为有限浮点观测；NaN/Inf 需要在进入插件前编码。Wasserstein 偏移与错误观测并不等价，真实的新状态也可能触发修复；较早历史本身被污染时 anchor 也可能不可靠。近期短块的均值、方差估计存在采样误差。阈值、anchor 与修复收益需要在目标数据的干净及近期扰动场景中评估。

梯度依赖检测增加一次 backbone 前向和两次输入梯度探测，推理时也有额外成本。现有实现未执行真实数据集上的完整性能对比，不应据此声称预测精度、鲁棒性或论文原创性已被验证。

参考思路来自用户提供的 TameR（最近数据偏置与梯度重要性分析）和 SEER（低质量块识别与替换）。RADAR 按当前方案修复观测数据；SEER 的参考替换机制作用于表示空间。TameR 附录的分析梯度使用预测 MSE，本插件使用预测输出的雅可比估计，以支持不提供未来真值的部署。

本地功能测试：

```bash
python -m unittest discover -s tests -v
```
