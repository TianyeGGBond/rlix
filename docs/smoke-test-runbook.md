# M11.2 Dual-Pipeline Smoke Test Runbook

> 2026-06-26 verified end-to-end. Targets `radixark/miles_latest` Docker image + Vast.ai environment.

---

## One-Liner Prompt (for AI Agents)

```
Rent a 4x RTX 5090 (32GB) machine on Vast.ai using the radixark/miles_latest template.
SSH in and run the M11.2 dual-pipeline smoke test:

1. Set up miles (note: branch is on rlops/miles, NOT radixark/miles):
   cd /root/miles
   git remote add rlops https://github.com/rlops/miles.git
   git fetch rlops zhenyu/m11-mvp-test
   git checkout -b zhenyu/m11-mvp-test rlops/zhenyu/m11-mvp-test
   pip install -e . --no-deps

2. Set up rlix:
   cd /root && git clone https://github.com/rlops/rlix.git
   cd rlix && git checkout zhenyu/miles-mvp-e2e
   pip install -e . --no-deps

3. Install ROLL dependency (required by rlix):
   pip install "roll @ git+https://github.com/rlops/ROLL.git" --no-deps

4. Download model and datasets:
   hf download Qwen/Qwen2.5-0.5B --local-dir /root/Qwen2.5-0.5B
   hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
   hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024

5. Convert checkpoint:
   cd /root/miles
   source scripts/models/qwen2.5-0.5B.sh
   PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
       ${MODEL_ARGS[@]} --hf-checkpoint /root/Qwen2.5-0.5B --save /root/Qwen2.5-0.5B_torch_dist

6. Apply SGLang compatibility patches (see "Known Issues" section below)

7. Run smoke test:
   mkdir -p /root/logs
   SCRIPT=/root/rlix/scripts/run_smoke_dual.sh \
   SILENCE_LIMIT=900 RUN_LIMIT=3600 LOG=/root/logs/run.log \
   bash /root/rlix/scripts/run_smoke_with_watchdog.sh

8. Verify success: last line of tail /root/logs/run.log should be EXIT_CODE=0

Topology (controlled by env vars in run_smoke_dual.sh):
  P1: train=[0], infer=[0,1,2]
  P2: train=[3], infer=[1,2,3]
  overlap=[1,2]
Minimal training steps: --num-rollout 2
```

---

## Hardware Requirements

| Item | Requirement | Notes |
|------|-------------|-------|
| GPU | 4x ≥32GB (RTX 5090 / A40) | 12GB RTX 3060 will OOM |
| Template | `radixark/miles_latest` | [Template link](https://cloud.vast.ai?ref_id=93591&template_id=aad3af111541bdb77d4807f3e2a7a81b) |
| Container Size | ≥100GB | Image itself is ~30-50GB |
| Reliability | ≥99% | Lower values often fail to start |

---

## Full Steps

### Step 1: Rent a Machine

Open the template link, filter for 4x RTX 5090 or 4x A40, pick a machine with reliability ≥99%, and click RENT.

### Step 2: SSH In

```bash
ssh -p <PORT> root@<IP> -L 8080:localhost:8080
```

### Step 3: Set Up miles

The image ships with miles from `radixark/miles`, but the rlix-mode code is on the `zhenyu/m11-mvp-test` branch of `rlops/miles`:

```bash
cd /root/miles
git remote add rlops https://github.com/rlops/miles.git
git fetch rlops zhenyu/m11-mvp-test
git checkout -b zhenyu/m11-mvp-test rlops/zhenyu/m11-mvp-test
pip install -e . --no-deps
```

### Step 4: Set Up rlix

```bash
cd /root
git clone https://github.com/rlops/rlix.git
cd rlix
git checkout zhenyu/miles-mvp-e2e
pip install -e . --no-deps
```

### Step 5: Install ROLL

rlix depends on the `roll` package (declared in `pyproject.toml`). Since `--no-deps` skipped it, install separately:

```bash
pip install "roll @ git+https://github.com/rlops/ROLL.git" --no-deps
```

### Step 6: Download Model + Datasets

```bash
hf download Qwen/Qwen2.5-0.5B --local-dir /root/Qwen2.5-0.5B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
```

### Step 7: Convert Checkpoint

```bash
cd /root/miles
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-0.5B \
    --save /root/Qwen2.5-0.5B_torch_dist
```

### Step 8: Apply SGLang Compatibility Patches

The SGLang version in the Docker image (0.5.14.dev31) is newer than what the miles branch expects, with two API incompatibilities. Edit `/root/miles/miles/backends/sglang_utils/sglang_engine.py`:

**Patch 1: Weight update requires begin/end session wrapping**

Find the import block in the `_route_update_weights_from_cpu_bucket` function and replace with:

```python
        from sglang.srt.entrypoints.http_server import _global_state
        from sglang.srt.managers.io_struct import (
            BeginWeightUpdateReqInput,
            EndWeightUpdateReqInput,
            UpdateWeightsFromTensorReqInput,
        )
        from sglang.srt.utils import MultiprocessingSerializer
```

Find the `UpdateWeightsFromTensorReqInput` construction in the same function and change `flush_cache=True` to `flush_cache=False`:

```python
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
            load_format=None,
            flush_cache=False,
        )
```

Find the `update_weights_from_tensor` call and wrap with begin/end:

```python
        try:
            await _global_state.tokenizer_manager.begin_weight_update(
                BeginWeightUpdateReqInput(), None
            )
            success, message = await _global_state.tokenizer_manager.update_weights_from_tensor(
                obj, None
            )
            await _global_state.tokenizer_manager.end_weight_update(
                EndWeightUpdateReqInput(), None
            )
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"update_weights_from_tensor: {exc!r}"},
            )
```

**Patch 2: flush_cache fault tolerance**

Find the `def flush_cache(self)` method and replace the `raise TimeoutError(...)` at the end with a warning:

```python
        else:
            logger.warning("flush_cache timed out after 60 attempts, proceeding anyway")
```

Also add 400 status code handling inside the retry loop:

```python
                if response.status_code == 400:
                    logger.info(f"flush_cache returned 400 (attempt {attempt}), retrying in 1s...")
                    time.sleep(1)
                    continue
```

### Step 9: Run the Smoke Test

```bash
mkdir -p /root/logs
SCRIPT=/root/rlix/scripts/run_smoke_dual.sh \
SILENCE_LIMIT=900 RUN_LIMIT=3600 LOG=/root/logs/run.log \
bash /root/rlix/scripts/run_smoke_with_watchdog.sh
```

Expected runtime: **10-15 minutes**.

### Step 10: Verify Success

```bash
tail -1 /root/logs/run.log
# Should output: EXIT_CODE=0
```

Full success criteria:
- `EXIT_CODE=0`
- `mp1 training loop complete pipeline_id=miles_<uuid>`
- `mp2 training loop complete pipeline_id=miles_<uuid>`
- `shutdown_hard complete` x 2
- rollout_id=0 and rollout_id=1 step1-step4 all completed (both pipelines)
- 0 occurrences of `Traceback` / `KeyError` / `Failed to resolve actor`

---

## Known Issues & Troubleshooting

| # | Issue | Root Cause | Solution |
|---|-------|------------|----------|
| 1 | RTX 3060 (12GB) OOMs during miles standalone training | `backward_step` needs 2.03GB but only 1.93GB free. 12GB insufficient for shared inference+training | Use ≥32GB GPUs (RTX 5090 / A40) |
| 2 | Vast.ai instance fails to start (error state) | Host disk too small for image, network timeout, or unstable host | Pick machines with reliability ≥99%, try multiple hosts |
| 3 | Can't find `zhenyu/m11-mvp-test` branch in miles | Branch is on `rlops/miles` (team fork), not on `radixark/miles` (upstream, pre-installed in image) | `git remote add rlops https://github.com/rlops/miles.git` then fetch |
| 4 | `ModuleNotFoundError: No module named 'roll'` | rlix depends on ROLL; `pip install -e . --no-deps` skipped all dependencies | Install separately: `pip install "roll @ git+https://github.com/rlops/ROLL.git" --no-deps` |
| 5 | `AssertionError: update_weights_from_tensor requires an open begin_weight_update session` | Docker image SGLang (0.5.14.dev31) added session-based weight update API; miles branch not adapted | Patch 1: wrap with `begin_weight_update` / `end_weight_update` in `_route_update_weights_from_cpu_bucket` |
| 6 | `TimeoutError: Timeout while flushing cache` (`GET /flush_cache` returns 400 repeatedly) | Newer SGLang briefly rejects flush requests after weight update state transitions | Patch 2: handle 400 retries in flush_cache + change timeout to `logger.warning` instead of `raise`; also set `flush_cache=False` in `UpdateWeightsFromTensorReqInput` |
| 7 | `vastai_setup.md` doesn't apply to rlix smoke test | That doc was written for NeMo/vLLM; not updated after switching to miles (SGLang) | Use this runbook instead |
| 8 | `/root/logs/` doesn't exist, training log write fails | Directory not pre-created in image | `mkdir -p /root/logs` before running the script |
| 9 | NCCL communication failure (first attempt with generic PyTorch template + `setup_env.sh`) | Machine had PCIe 3.0 8x, no NVLink, no CUDA IPC support | Use team template `radixark/miles_latest`; pick machines with NVLink or at least PCIe 4.0 |

---

## Branch Info

| Repo | Branch | Purpose |
|------|--------|---------|
| `rlops/rlix` | `zhenyu/miles-mvp-e2e` | rlix control plane + F1-F12 |
| `rlops/miles` | `zhenyu/m11-mvp-test` | miles-side rlix-mode integration (examples/rlix/) |
| `rlops/ROLL` | `main` | Scheduling framework dependency for rlix |

---

## After the Test

- **Stop** the instance: preserves environment, only charges disk fee ($0.01-0.03/hr)
- **Destroy** the instance: fully released, no charges but all data lost
- Record results in the Google Sheet `miles review progress`

---
---

# 中文版 / Chinese Version

# M11.2 双 Pipeline Smoke Test 运行手册

> 2026-06-26 实测跑通记录。适用于 `radixark/miles_latest` Docker 镜像 + Vast.ai 环境。

---

## 一句话 Prompt（给 AI Agent 用）

```
在 Vast.ai 上用 radixark/miles_latest 模板租一台 4x RTX 5090 (32GB) 机器。
SSH 进去后，按以下步骤跑 M11.2 dual-pipeline smoke test：

1. 设置 miles（注意：分支在 rlops/miles，不是 radixark/miles）：
   cd /root/miles
   git remote add rlops https://github.com/rlops/miles.git
   git fetch rlops zhenyu/m11-mvp-test
   git checkout -b zhenyu/m11-mvp-test rlops/zhenyu/m11-mvp-test
   pip install -e . --no-deps

2. 设置 rlix：
   cd /root && git clone https://github.com/rlops/rlix.git
   cd rlix && git checkout zhenyu/miles-mvp-e2e
   pip install -e . --no-deps

3. 安装 ROLL 依赖（rlix 需要）：
   pip install "roll @ git+https://github.com/rlops/ROLL.git" --no-deps

4. 下载模型和数据集：
   hf download Qwen/Qwen2.5-0.5B --local-dir /root/Qwen2.5-0.5B
   hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
   hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024

5. 转换 checkpoint：
   cd /root/miles
   source scripts/models/qwen2.5-0.5B.sh
   PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
       ${MODEL_ARGS[@]} --hf-checkpoint /root/Qwen2.5-0.5B --save /root/Qwen2.5-0.5B_torch_dist

6. 应用 SGLang 兼容性补丁（见下方"已知问题"节）

7. 跑 smoke test：
   mkdir -p /root/logs
   SCRIPT=/root/rlix/scripts/run_smoke_dual.sh \
   SILENCE_LIMIT=900 RUN_LIMIT=3600 LOG=/root/logs/run.log \
   bash /root/rlix/scripts/run_smoke_with_watchdog.sh

8. 验证成功：tail /root/logs/run.log 最后一行应为 EXIT_CODE=0

拓扑配置（由 run_smoke_dual.sh 内的环境变量控制）：
  P1: train=[0], infer=[0,1,2]
  P2: train=[3], infer=[1,2,3]
  overlap=[1,2]
训练步骤最小：--num-rollout 2
```

---

## 硬件要求

| 项目 | 要求 | 说明 |
|------|------|------|
| GPU | 4x ≥32GB（RTX 5090 / A40） | 12GB RTX 3060 会 OOM |
| 模板 | `radixark/miles_latest` | [模板链接](https://cloud.vast.ai?ref_id=93591&template_id=aad3af111541bdb77d4807f3e2a7a81b) |
| Container Size | ≥100GB | 镜像本身约 30-50GB |
| Reliability | ≥99% | 低于此值启动容易失败 |

---

## 完整步骤

### Step 1: 租机器

打开模板链接，筛选 4x RTX 5090 或 4x A40，选 reliability ≥99% 的机器，点 RENT。

### Step 2: SSH 进入

```bash
ssh -p <PORT> root@<IP> -L 8080:localhost:8080
```

### Step 3: 设置 miles

镜像里预装的 miles 来自 `radixark/miles`，但 rlix-mode 代码在 `rlops/miles` 的 `zhenyu/m11-mvp-test` 分支：

```bash
cd /root/miles
git remote add rlops https://github.com/rlops/miles.git
git fetch rlops zhenyu/m11-mvp-test
git checkout -b zhenyu/m11-mvp-test rlops/zhenyu/m11-mvp-test
pip install -e . --no-deps
```

### Step 4: 设置 rlix

```bash
cd /root
git clone https://github.com/rlops/rlix.git
cd rlix
git checkout zhenyu/miles-mvp-e2e
pip install -e . --no-deps
```

### Step 5: 安装 ROLL

rlix 依赖 `roll` 包（在 `pyproject.toml` 中声明），`--no-deps` 跳过了它，需要单独装：

```bash
pip install "roll @ git+https://github.com/rlops/ROLL.git" --no-deps
```

### Step 6: 下载模型 + 数据集

```bash
hf download Qwen/Qwen2.5-0.5B --local-dir /root/Qwen2.5-0.5B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
```

### Step 7: 转换 checkpoint

```bash
cd /root/miles
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-0.5B \
    --save /root/Qwen2.5-0.5B_torch_dist
```

### Step 8: 应用 SGLang 兼容性补丁

Docker 镜像中的 SGLang (0.5.14.dev31) 比 miles 分支代码期望的版本更新，有两处 API 不兼容。需要修改 `/root/miles/miles/backends/sglang_utils/sglang_engine.py`：

**补丁 1：weight update 需要 begin/end session 包裹**

找到 `_route_update_weights_from_cpu_bucket` 函数中的 import 块，替换为：

```python
        from sglang.srt.entrypoints.http_server import _global_state
        from sglang.srt.managers.io_struct import (
            BeginWeightUpdateReqInput,
            EndWeightUpdateReqInput,
            UpdateWeightsFromTensorReqInput,
        )
        from sglang.srt.utils import MultiprocessingSerializer
```

找到同函数中的 `UpdateWeightsFromTensorReqInput` 构造，将 `flush_cache=True` 改为 `flush_cache=False`：

```python
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
            load_format=None,
            flush_cache=False,
        )
```

找到 `update_weights_from_tensor` 调用，加上 begin/end 包裹：

```python
        try:
            await _global_state.tokenizer_manager.begin_weight_update(
                BeginWeightUpdateReqInput(), None
            )
            success, message = await _global_state.tokenizer_manager.update_weights_from_tensor(
                obj, None
            )
            await _global_state.tokenizer_manager.end_weight_update(
                EndWeightUpdateReqInput(), None
            )
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"update_weights_from_tensor: {exc!r}"},
            )
```

**补丁 2：flush_cache 容错**

找到 `def flush_cache(self)` 方法，将末尾的 `raise TimeoutError(...)` 改为 warning：

```python
        else:
            logger.warning("flush_cache timed out after 60 attempts, proceeding anyway")
```

同时在循环里加上 400 状态码处理：

```python
                if response.status_code == 400:
                    logger.info(f"flush_cache returned 400 (attempt {attempt}), retrying in 1s...")
                    time.sleep(1)
                    continue
```

### Step 9: 跑 Smoke Test

```bash
mkdir -p /root/logs
SCRIPT=/root/rlix/scripts/run_smoke_dual.sh \
SILENCE_LIMIT=900 RUN_LIMIT=3600 LOG=/root/logs/run.log \
bash /root/rlix/scripts/run_smoke_with_watchdog.sh
```

预计运行 **10-15 分钟**。

### Step 10: 验证成功

```bash
tail -1 /root/logs/run.log
# 应输出: EXIT_CODE=0
```

完整成功标准：
- `EXIT_CODE=0`
- `mp1 training loop complete pipeline_id=miles_<uuid>`
- `mp2 training loop complete pipeline_id=miles_<uuid>`
- `shutdown_hard complete` x 2
- rollout_id=0 和 rollout_id=1 的 step1-step4 全部完成（两个 pipeline）
- 0 个 `Traceback` / `KeyError` / `Failed to resolve actor`

---

## 已知问题 & 踩坑记录

| # | 问题 | 根因 | 解决方案 |
|---|------|------|----------|
| 1 | RTX 3060 (12GB) 跑 miles 独立训练 OOM | `backward_step` 时 GPU 只剩 1.93GB，需要 2.03GB。12GB 不够推理+训练共享 | 换 ≥32GB GPU（RTX 5090 / A40） |
| 2 | Vast.ai 实例启动失败（error 状态） | 主机磁盘不够拉镜像、网络超时、主机不稳定 | 选 reliability ≥99% 的机器，多试几台 |
| 3 | miles 的 `zhenyu/m11-mvp-test` 分支找不到 | 分支在 `rlops/miles`（团队 fork），不在镜像预装的 `radixark/miles`（上游 repo） | `git remote add rlops https://github.com/rlops/miles.git` 后 fetch |
| 4 | `ModuleNotFoundError: No module named 'roll'` | rlix 依赖 ROLL 包，`pip install -e . --no-deps` 跳过了所有依赖 | 单独 `pip install "roll @ git+https://github.com/rlops/ROLL.git" --no-deps` |
| 5 | `AssertionError: update_weights_from_tensor requires an open begin_weight_update session` | Docker 镜像里的 SGLang (0.5.14.dev31) 新增了 session-based weight update API，miles 分支代码未适配 | 补丁 1：在 `_route_update_weights_from_cpu_bucket` 中加 `begin_weight_update` / `end_weight_update` 包裹 |
| 6 | `TimeoutError: Timeout while flushing cache`（`GET /flush_cache` 返回 400 十几次直到超时） | 新版 SGLang 在 weight update 状态转换后短暂拒绝 flush 请求 | 补丁 2：flush_cache 中处理 400 重试 + 超时改 `logger.warning` 而非 `raise`；同时将 `UpdateWeightsFromTensorReqInput` 的 `flush_cache=True` 改为 `False` |
| 7 | `vastai_setup.md` 不适用于 rlix smoke test | 该文档是给 NeMo/vLLM 写的，切换到 miles (SGLang) 后未更新 | 用本 runbook 替代 |
| 8 | `/root/logs/` 目录不存在导致训练日志写入失败 | 镜像里没有预创建该目录 | 跑脚本前 `mkdir -p /root/logs` |
| 9 | 第一次尝试（通用 PyTorch 模板 + `setup_env.sh`）NCCL 通信失败 | 机器 PCIe 3.0 8x，无 NVLink，无 CUDA IPC 支持 | 使用团队模板 `radixark/miles_latest`，选支持 NVLink 或至少 PCIe 4.0 的机器 |

---

## 分支信息

| Repo | 分支 | 说明 |
|------|------|------|
| `rlops/rlix` | `zhenyu/miles-mvp-e2e` | rlix 控制平面 + F1-F12 |
| `rlops/miles` | `zhenyu/m11-mvp-test` | miles 侧 rlix-mode 接入（examples/rlix/） |
| `rlops/ROLL` | `main` | rlix 的调度框架依赖 |

---

## 跑完后

- **Stop** 实例：保留环境，只收磁盘费（$0.01-0.03/hr）
- **Destroy** 实例：完全释放，不收费但数据全没
- 在 Google Sheet `miles review progress` 中记录
