# Contributing to SpecLoc

感谢你改进 SpecLoc。项目优先接受能够提高可复现性、数据审计、实验可解释性或测试覆盖的
变更。

## 开发环境

```bash
git clone https://github.com/DexZane/SpecLoc.git
cd SpecLoc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

无 NVIDIA GPU 时使用：

```bash
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements/dev.txt
```

## 提交前检查

```bash
make lint
make test
specloc doctor rsod --skip-data --allow-cpu
```

新增功能必须包含能够覆盖用户可见行为的测试。涉及数据划分、指标或统计合同的改动，必须
说明是否改变历史实验的可比性。

## Pull request

1. 从 `main` 创建主题分支；
2. 保持提交聚焦，不混入数据集、checkpoint 或实验输出；
3. 更新 README、运行手册或 CHANGELOG 中受影响的部分；
4. 在 PR 中列出验证命令和结果；
5. 确保 GitHub Actions 通过。

## 数据与许可证

不要提交 RSOD、AI-TOD-v2 或其他第三方数据。新增数据适配器时，应记录上游来源、许可、
校验和、划分规则和防泄漏检查。
