# 数据文件说明

此目录用于放置赛题官方附件。原始数据未包含在仓库中，请自行放入：

```text
附件1.xlsx
附件2.xlsx
附件3.xlsx
附件4.xlsx
附件5/
├─ result1.xlsx
├─ result2.xlsx
├─ result3.xlsx
├─ result4-2.xlsx
└─ result4-3.xlsx
```

如果附件位于其他位置，可以设置环境变量 `CUMCM_ATTACHMENT_DIR`，其值应为包含 `附件1.xlsx`～`附件4.xlsx` 和 `附件5` 目录的路径。
