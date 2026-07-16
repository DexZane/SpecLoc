# Changelog

本项目遵循面向实验复现的版本记录。尚未发布到 PyPI 的版本以 Git tag 和 commit 为准。

## 0.8.0 - 2026-07-16

- 新增 CUDA 12.1、CUDA 11.8、CPU 和开发依赖 profiles；
- 固定 Setuptools 80.10.2，兼容 PyTorch 2.1.2 的 `pkg_resources` 导入；
- 新增 `specloc` 统一命令，覆盖信息检查、数据校验、训练前验收、训练和评估；
- 默认单卡训练收敛为 `specloc train rsod`；
- CI 改为直接验证公开 requirements 安装入口；
- 增加贡献指南、Issue/PR 模板和 lint 合同。

## 0.7.0 - 2026-07-16

- 项目、包和元数据统一更名为 SpecLoc；
- 新增经审计的 RSOD 合并数据配置、校验器和独立 train/val/test 流程；
- 保留 AI-TOD-v2 正式基线与频谱信息量 Gate 1。
