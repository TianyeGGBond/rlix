# 博客大纲：大规模 Agentic RL 的局部时分复用（Partial Time-Sharing）优化实践

## 1. 核心挑战：Agentic RL 中的“长尾与空泡”
*   **Agentic RL 系统负载特征**：
    *   **长时、多轮交互**：如 GPT-5.1-Codex 般持续数小时的任务执行时长，增加了采样过程中的不确定性。
    *   **长尾分布**：分布式环境下，极少数复杂样本（Stragglers）拖慢整个采样阶段的完成时间。
*   **核心痛点：落后者效应 (Straggler Effect)**：
    *   **资源空泡的形成**：先行完成的节点受限于同步机制，无法提前进入下一阶段，导致严重的分布式算力闲置。

## 2. 调度策略的演进与对比
*   **策略 1：同步共置（Synchronous Co-location / Baseline）**
    *   **机制**：串行执行。所有 GPU 先进行 **Rollout Phase** (Agentic Rollout)，再进行 **Training Phase** (Trainer)。
    *   **缺陷**：阶段内资源闲置；Rollout Phase 与 Training Phase 无重叠。
*   **策略 2：异步分离（Asynchronous Separation / Static Pipeline）**
    *   **机制**：静态分配 GPU 资源池，分别用于 **Worker Group** (Rollout) 和 **Training Group** (Training)。
    *   **缺陷**：**"双边空泡"问题**；无法应对采样量的动态波动。
*   **策略 3：局部时分复用（Partial Time-Sharing - 本方案）**
    *   **核心洞察**：随着 **Trajectory** 生成完成，**Prompt Batch** 的积压减少，**Agentic Rollout Worker** 的负载随时间自然下降（出现需求波谷）。
    *   **核心逻辑**：将 **Training Role** 的执行时间平移至 **Rollout Phase** 的“需求波谷期”，实现资源需求与供给的完美拟合。

### 技术特性对比矩阵
| 维度 | 同步共置 | 异步分离 | 局部时分复用 (本方案) |
| :--- | :--- | :--- | :--- |
| **并行触发时机** | **Rollout Phase** 100% 完成后 | 固定流水线 | **Rollout Phase** 趋势波谷时 (按需) |
| **资源利用率** | 极低 | 中 (因双边空泡闲置) | **极高 (重叠掩护 Straggler 耗时)** |
| **分布式复杂度**| 低 | 中 | **高 (动态 Job Coordinator 调度)** |

## 3. 实现细节解析：确保逻辑闭环的三个关键
### 3.1 动态流水线的两个阶段
1.  **全力采样 (Full Rollout Phase)**：启动阶段 **GPU Cluster** 100% 投入采样，利用规模效应快速解决高并发的 **Prompt Batch**。
2.  **局部重叠 (Partial Overlap Phase)**：
    *   **Shrink 操作**：**Job Coordinator** 自动识别采样完成度，剥离 128 个 **DP Worker** 转入 **Training Phase**，仅保留 32 个 **DP Worker** 处理长尾 **Trajectory**。
    *   **并行加速**：**Training Role**与长尾 **Agentic Rollout** 在物理上并行执行，训练耗时被长尾采样时长掩护。
    *   **Expand 操作**：训练结束即回收 128 个 **DP Worker**，确保下一轮 **Rollout Phase** 峰值拥有全量算力。

### 3.2 消除技术断层的工程设计
*   **状态迁移 (State Migration / Migration by Abort)**：被剥离节点的 in-flight **Prompt Batch** 需迁移至活跃节点，确保持续交互不因资源剥离而中断。
*   **分布式原子性控制**：确保 **Job Coordinator** 内 Shrink/Expand 操作的一致性，防止状态不合拍导致的死锁。
*   **显存动态换出 (Swap / Offload)**：切换前自动释放 **Inference Engine** 的模型权重（Asset），解决大规模模型共存的显存限制。

## 4. 实验证明：性能与损耗的终极博弈
*   **实验设置**：160x L20x **GPU Cluster**，SWE-Agentic 真实负载，Qwen3-235B 模型。
*   **证明点 1：吞吐量与耗时**
    *   **Rollout Phase** 耗时从 ~5000s 降至 **<2500s**。证明：重叠长尾确实能消灭“波谷闲置”。
*   **证明点 2：任务稳定性 (Zero Timeouts)**
    *   Baseline 因 **GPU Pool** 静态配置不足（32卡）导致采样极慢并触发环境超时；本方案初始即投入 160 卡。证明：前期全量资源投入能有效对抗采样长尾，消除超时风险。
*   **证明点 3：切换损耗的可控性博弈**
    *   **Swap** (4s) + **Sync Weights** (2s) = **6s**。相对于小时级的训练节省，6s 损耗微乎其微。证明：动态调度是极高 ROI 的优化手段。

## 5. 结论
*   局部时分复用优化通过构建“动态资源池”，成功将分布式系统的“劣势”（长尾等待）转化为“优势”（训练窗口）。
*   本方案为超大规模 Agentic RL 在有限硬件资源下的极致性能输出提供了标准范式。
