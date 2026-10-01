# RoboDojo MEM：DM05、π0.5、OpenWAM 的纯 RGB IKV 实验

设计参照 [genalyu/ikv](https://github.com/genalyu/ikv) 的
[RGB motion](https://github.com/genalyu/ikv/blob/main/RGB_ONLY.md) 与
[indexed KV](https://github.com/genalyu/ikv/blob/main/docs/INDEXED_KV.md)：
先比较相邻真实 RGB 帧，找到发生变化的视觉 patch；保留原模型产生的视觉
embedding/K/V；另外用冻结 DINOv2 为 patch 建立语义索引。索引只用于筛选，
不参与模型的 Q/K/V 投影。这里没有深度、相机位姿、触觉或 NeoForce。

## 原版逻辑与本次改动

| 模型 | 原版 | 本次纯 RGB IKV 路径 | 可预期的机制效果 |
| --- | --- | --- | --- |
| DM05-MEM | RoboDojo MEM 把最多 20 帧历史图像各缩成 4×4 视觉 token，与当前图像和文本一起构成 VLM prefix；动作专家对固定 prefix 做流匹配采样。 | `opendm/playground/dm05_mem_ikv_rgb_robodojo_cover_blocks.py` 对历史帧逐 patch 比较相邻 RGB，也比较最后一帧历史与当前真实 RGB。第一帧完整进入，之后只有变化 patch 候选进入；当前真实帧的 DINOv2 特征是评分基准。容量不足时保留高分 top-k，再从剩余候选随机抽取。batch=1 推理时按同一位置表压缩各层 prefix K/V。 | 历史中未变化的视觉 token 不再占据可见 prefix；推理缓存列数下降。当前视觉和动作输出接口不变。 |
| π0.5 | SigLIP/PaliGemma 编码当前三路 RGB 与语言，动作专家采样；原配置没有 RoboDojo 历史图像 memory token。 | `pi05_ikv_rgb_robodojo` 增加按时间排列的 `memory_000_rgb` 等图像。每张图像仍由原 SigLIP 生成 patch embedding；相邻历史帧及 `t-1 → t` 的 RGB 差确定候选，当前真实帧 DINOv2 特征是评分基准。JAX/PyTorch 在 PaliGemma prefill 前聚拢选中 token。 | 额外历史视觉证据进入动作条件，历史 prefix 被容量限制。JAX 的固定 shape 可留下不可见 padding。 |
| OpenWAM | Wan TI2V 先把条件 RGB 编成因果 VAE latent，视频分支和动作分支再走各自或联合的去噪；IDM 第二阶段会预填视频 K/V。生成的预测视频没有跨请求持久缓存。 | 策略保存真实 RGB 历史，并在左侧重复最早一帧，使最新真实帧落在 VAE 的 0,4,8,… 时间端点。对相邻原始 RGB 求差，再把每段的变化合到对应 Wan patch；DINOv2 是独立索引。位置表每次请求只计算一次，联合注意力与 IDM 视频预填在每层按同一表收缩 K/V；视频生成 query 仍保持完整。 | 真实历史的静止 patch 被遮蔽或物理压缩，IDM 的 IKV 参数已能到达视频 backbone。预测视频仍按原版生成，并在请求结束时释放。 |

### 变化 patch 与压缩规则

1. RGB 先归一化到 `[0,1]`。在相邻真实帧（包括历史末帧到当前帧）上计算逐像素的平均绝对通道差，**先取绝对差，再汇聚到模型的 patch 网格**。首个有效观测提供完整基准帧；后续 patch 仅在差值超过阈值时成为候选。DM05 另从数据入口传原始 RGB，避免误把已标准化的视觉编码输入当成 RGB；训练时头部相机当前帧和历史帧复用同一组图像增强参数。π0.5 的变化检测使用增强前当前 RGB，防止增强造成伪运动。
2. 每个候选对应原有视觉 token 的时间和网格位置。冻结 DINOv2 的 patch 特征单独保存/传递，用候选与最近真实观测 `t0` 的 DINO patch 做最大非负余弦相似度。时间项随 `|t-t0|` 衰减。DINO 不混入视觉 embedding 或 KV 值。
3. 容量不足时，按视觉差、DINO 相似度、时间邻近度排序并保护 `top_k`；其它候选通过可复现的随机次序补足容量。未变化的 patch **不会为了填满容量而重新加入**。各层使用同一选择表。
4. 底层选择函数还接受查询使用量和动作重复度数组；目前三个模型的注意力循环没有提供真实的跨请求统计，因此这些项在 RoboDojo 入口中尚未启用。触觉权重、NeoForce 均不存在。

这三个基线尚未拥有 N0-TWAM 的全局跨请求流式 K/V 池：DM05 和 π0.5 每次请求重算 prefix，OpenWAM 每次请求重新准备视频条件。因而这里的“KV 压缩”是**请求内**的 prefix/联合注意力压缩；不能把它当作跨请求复用旧 K/V 或完整复刻参考仓库的 query-mass/动作 token 保留机制。实际任务效果尚需 RoboDojo MEM 评测。

### DINOv2 配置

仅从本地权重加载 DINOv2（`local_files_only=True`），不会自动下载。
RoboDojo 的三个 IKV 入口默认要求语义索引；运行前设置本地 checkpoint，
也可提供与 patch 对齐的预计算特征。

| 模型 | 本地 checkpoint | 预计算特征 |
| --- | --- | --- |
| DM05 | `--model-config.ikv-dino-model-path ./checkpoints/dinov2-base` | 模型调用里的 `ikv_dino_features: [历史帧,16,D]`，另有当前真实 RGB 的 `ikv_reference_dino: [batch,16,D]` |
| π0.5 | `LeRobotAlohaIKVDataConfig(ikv_dino_model_path=...)` | 数据记录里的 `ikv_dino_features: [历史帧,256,D]`，另有当前真实 RGB 的 `ikv_reference_dino: [256,D]` |
| OpenWAM | `inference.ikv_dino_model_path` | 观测字典里的 `ikv_dino_features: [历史帧,H,W,D]` |

只有做无 DINO 消融时才关闭各入口的 `ikv_require_dino`。π0.5 的
预计算特征需对应标准 224×224、16×16 patch 网格；OpenWAM 会把 16×16
DINO 索引池化到实际 Wan patch 网格。

## RoboDojo 入口

DM05-MEM 沿用 `opendm/docs/en/dm05_robodojo.md` 的数据注册、20 帧历史、
动作归一化和 checkpoint。例如：

```bash
cd opendm
script/dm05_launcher.sh \
  --exp playground/dm05_mem_ikv_rgb_robodojo_cover_blocks.py \
  --task train --nproc_per_node 8 \
  --data-config.dataset-name robodojo_sim_cover_blocks \
  --model-config.model-name-or-path ./checkpoints/DM05-MEM \
  --model-config.ikv-dino-model-path ./checkpoints/dinov2-base
```

π0.5 的 `pi05_ikv_rgb_robodojo` 需要 LeRobot 记录中包含
`observation.images.{top,left_wrist,right_wrist}`、按时间排列的
`observation.history_images`、`observation.state` 和 `action`。输入适配器
`AlohaIKVRGBInputs` 生成固定长度的 `memory_000_rgb` 等键。该数据转换
尚未随本仓库生成；训练与 RoboDojo 推理必须使用相同的历史图像契约。

OpenWAM 在 `OpenWAM/configs/deploy.yaml` 启用 `inference.ikv_rgb_enabled`
并设置 DINO checkpoint；该路径要求 Wan TI2V 模型。策略 `reset()` 清空
RGB/DINO 历史。OpenWAM 训练加载器尚未扩展成长历史 clean prefix。

## 验证范围

CPU 上的 patch 选择、语义索引、K/V 列对齐测试及 Python 语法检查已通过。
本地缺少 JAX 与 albumentations，对应测试跳过；OpenWAM 缺少
`modelscope`，无法在本机装载完整模型运行推理。
此目录没有 DINO/模型权重、RoboDojo 数据或 GPU 模拟器，尚无成功率、
时延或显存数字。原版预测分支的 K/V 生命周期未改变：请求内使用，下一次
真实观测会重新建立条件，而不是把预测 K/V 作为真实记忆保存。
