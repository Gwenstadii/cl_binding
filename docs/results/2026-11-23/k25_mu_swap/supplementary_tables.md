# LF→Right 共同响应替换：完整补充表

每个条件、每条旧目标 n=20。成对要求同一初态复位执行 Left 和 Front 都正确。200 条历史完整复跑包含在 600 条内，不新增独立样本。

## 全部端点、读出与干预前事件

| 窗口 | 条件 | 目标 | 首达 | 终态 | 曾成功 | 抓取 | 首达分布 | 首次读出 | 全块读对回合 | 并列块 | 父速度场查询数 | 首块 Δd RMS 中位数 | 首块 μ 差 RMS 中位数 | 干预前首达 | 干预前抓取 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无 | 原生 F，晚期 μ | Left | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {} | None | 0 | 0 | None | None | {"none": 20} | 0 |
| 无 | 原生 F，晚期 μ | Front | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {} | None | 0 | 0 | None | None | {"none": 20} | 0 |
| 81–180 | 只换父 μ，父 μ | Left | 5 | 7 | 17 | 17 | {"front": 12, "left": 5, "none": 3} | {"A": 20} | 20 | 0 | 4000 | 0.0 | 0.8348397940397263 | {"none": 20} | 17 |
| 81–180 | 只换父 μ，父 μ | Front | 15 | 12 | 13 | 17 | {"front": 15, "none": 5} | {"B": 20} | 20 | 0 | 4000 | 0.0 | 0.8824487537145614 | {"none": 20} | 15 |
| 81–180 | K25 V＋，晚期 μ | Left | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0 | 0.042095350846648213 | None | {"none": 20} | 17 |
| 81–180 | K25 V＋，晚期 μ | Front | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0 | 0.034475832339376206 | None | {"none": 20} | 15 |
| 81–180 | K25 V＋，父 μ | Left | 6 | 5 | 13 | 17 | {"front": 11, "left": 6, "none": 3} | {"A": 20} | 20 | 0 | 4000 | 0.04659586353227496 | 0.8215797513723373 | {"none": 20} | 17 |
| 81–180 | K25 V＋，父 μ | Front | 12 | 11 | 11 | 17 | {"front": 12, "none": 8} | {"B": 20} | 20 | 0 | 4000 | 0.03679511900991202 | 0.9011844515800476 | {"none": 20} | 15 |
| 81–180 | K25 V−，晚期 μ | Left | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0 | 0.04073445154353976 | None | {"none": 20} | 17 |
| 81–180 | K25 V−，晚期 μ | Front | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0 | 0.035089040920138356 | None | {"none": 20} | 15 |
| 81–180 | K25 V−，父 μ | Left | 4 | 2 | 13 | 17 | {"front": 13, "left": 4, "none": 3} | {"A": 20} | 20 | 0 | 4000 | 0.04423412531614304 | 0.8636611998081207 | {"none": 20} | 17 |
| 81–180 | K25 V−，父 μ | Front | 17 | 6 | 11 | 17 | {"front": 17, "none": 3} | {"B": 20} | 20 | 0 | 4000 | 0.037931901030242444 | 0.8763814985752105 | {"none": 20} | 15 |
| 81–180 | 2.959 倍自身残差，晚期 μ | Left | 0 | 0 | 0 | 17 | {"back": 8, "front": 2, "none": 3, "right": 7} | {"A": 20} | 20 | 0 | 0 | 0.0 | 0.0 | {"none": 20} | 17 |
| 81–180 | 2.959 倍自身残差，晚期 μ | Front | 0 | 0 | 0 | 17 | {"none": 13, "right": 7} | {"B": 20} | 20 | 0 | 0 | 0.0 | 0.0 | {"none": 20} | 15 |
| 81–180 | 2.959 倍自身残差，父 μ | Left | 10 | 10 | 14 | 17 | {"front": 7, "left": 10, "none": 3} | {"A": 20} | 20 | 0 | 4000 | 0.0 | 0.8284620553255082 | {"none": 20} | 17 |
| 81–180 | 2.959 倍自身残差，父 μ | Front | 0 | 0 | 0 | 17 | {"none": 20} | {"B": 20} | 20 | 0 | 4000 | 0.0 | 0.9051978498697282 | {"none": 20} | 15 |
| 1–400 | 只换父 μ，父 μ | Left | 18 | 17 | 18 | 18 | {"left": 18, "none": 2} | {"A": 20} | 20 | 0 | 16000 | 0.0 | 0.4336071655154228 | {"none": 20} | 0 |
| 1–400 | 只换父 μ，父 μ | Front | 18 | 3 | 3 | 20 | {"front": 18, "left": 2} | {"B": 20} | 20 | 0 | 16000 | 0.0 | 0.44256215840578084 | {"none": 20} | 0 |
| 1–400 | K25 V＋，晚期 μ | Left | 0 | 0 | 0 | 18 | {"none": 2, "right": 18} | {"A": 20} | 20 | 0 | 0 | 0.022098172316327692 | None | {"none": 20} | 0 |
| 1–400 | K25 V＋，晚期 μ | Front | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0 | 0.020791955804452297 | None | {"none": 20} | 0 |
| 1–400 | K25 V＋，父 μ | Left | 17 | 16 | 17 | 17 | {"left": 17, "none": 3} | {"A": 20} | 20 | 0 | 16000 | 0.02201265562325716 | 0.4313294470310211 | {"none": 20} | 0 |
| 1–400 | K25 V＋，父 μ | Front | 14 | 7 | 8 | 20 | {"front": 14, "left": 1, "none": 5} | {"B": 20} | 20 | 0 | 16000 | 0.021446975925937296 | 0.44133013635873797 | {"none": 20} | 0 |
| 1–400 | K25 V−，晚期 μ | Left | 0 | 0 | 0 | 18 | {"none": 2, "right": 18} | {"A": 20} | 20 | 0 | 0 | 0.02115645674057305 | None | {"none": 20} | 0 |
| 1–400 | K25 V−，晚期 μ | Front | 0 | 0 | 0 | 18 | {"none": 2, "right": 18} | {"B": 20} | 20 | 0 | 0 | 0.02140827467665076 | None | {"none": 20} | 0 |
| 1–400 | K25 V−，父 μ | Left | 20 | 14 | 20 | 20 | {"left": 20} | {"A": 20} | 20 | 0 | 16000 | 0.022117076674476266 | 0.4349381163716316 | {"none": 20} | 0 |
| 1–400 | K25 V−，父 μ | Front | 9 | 1 | 1 | 18 | {"front": 9, "left": 9, "none": 2} | {"B": 20} | 20 | 0 | 16000 | 0.023016257863491773 | 0.4453593298792839 | {"none": 20} | 0 |
| 1–400 | 2.959 倍自身残差，晚期 μ | Left | 0 | 0 | 0 | 18 | {"back": 2, "front": 14, "none": 2, "right": 2} | {"A": 20} | 20 | 0 | 0 | 0.0 | 0.0 | {"none": 20} | 0 |
| 1–400 | 2.959 倍自身残差，晚期 μ | Front | 0 | 0 | 0 | 13 | {"none": 18, "right": 2} | {"B": 20} | 20 | 0 | 0 | 0.0 | 0.0 | {"none": 20} | 0 |
| 1–400 | 2.959 倍自身残差，父 μ | Left | 17 | 14 | 17 | 17 | {"left": 17, "none": 3} | {"A": 20} | 20 | 0 | 16000 | 0.0 | 0.4302658542990685 | {"none": 20} | 0 |
| 1–400 | 2.959 倍自身残差，父 μ | Front | 3 | 0 | 3 | 20 | {"front": 3, "none": 17} | {"B": 20} | 20 | 0 | 16000 | 0.0 | 0.44042355865240096 | {"none": 20} | 0 |

RMS 覆盖速度场 50 个预测动作槽的七个执行维度，首块内十次去噪取均值，再取二十回合中位数；不是机器人位移。每块最终执行前十个动作。旧 K25 参考未记录 μ 差时记 null／None，不解释成差值为零。干预前事件必须五个确认步均在干预前；窗口 1–400 的干预前集合为空。

## 相对原生 F 的逐初态转移

| 窗口 | 后条件 | 目标／joint | 首达获益／损失 | 终态获益／损失 | 前→后首达转移 | 获益／损失初态编号 |
| --- | --- | --- | --- | --- | --- | --- |
| 无 | 原生 F，晚期 μ | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 无 | 原生 F，晚期 μ | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 无 | 原生 F，晚期 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | 只换父 μ，父 μ | A | 5／0 | 7／0 | {"none->none": 3, "right->front": 12, "right->left": 5} | {"first_gained": [801, 806, 809, 814, 818], "first_lost": [], "terminal_gained": [801, 804, 806, 807, 808, 809, 819], "terminal_lost": []} |
| 81–180 | 只换父 μ，父 μ | B | 15／0 | 12／0 | {"none->none": 3, "right->front": 15, "right->none": 2} | {"first_gained": [800, 802, 804, 805, 806, 807, 808, 809, 810, 811, 812, 814, 817, 818, 819], "first_lost": [], "terminal_gained": [802, 804, 805, 807, 808, 809, 810, 811, 812, 814, 818, 819], "terminal_lost": []} |
| 81–180 | 只换父 μ，父 μ | joint | 4／0 | 5／0 | {} | {"first_gained": [806, 809, 814, 818], "first_lost": [], "terminal_gained": [804, 807, 808, 809, 819], "terminal_lost": []} |
| 81–180 | K25 V＋，晚期 μ | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | K25 V＋，晚期 μ | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | K25 V＋，晚期 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | K25 V＋，父 μ | A | 6／0 | 5／0 | {"none->none": 3, "right->front": 11, "right->left": 6} | {"first_gained": [801, 802, 806, 809, 814, 818], "first_lost": [], "terminal_gained": [807, 808, 809, 814, 817], "terminal_lost": []} |
| 81–180 | K25 V＋，父 μ | B | 12／0 | 11／0 | {"none->none": 3, "right->front": 12, "right->none": 5} | {"first_gained": [800, 804, 805, 807, 808, 809, 811, 812, 814, 817, 818, 819], "first_lost": [], "terminal_gained": [804, 805, 807, 808, 809, 811, 812, 814, 817, 818, 819], "terminal_lost": []} |
| 81–180 | K25 V＋，父 μ | joint | 3／0 | 5／0 | {} | {"first_gained": [809, 814, 818], "first_lost": [], "terminal_gained": [807, 808, 809, 814, 817], "terminal_lost": []} |
| 81–180 | K25 V−，晚期 μ | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | K25 V−，晚期 μ | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | K25 V−，晚期 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | K25 V−，父 μ | A | 4／0 | 2／0 | {"none->none": 3, "right->front": 13, "right->left": 4} | {"first_gained": [801, 806, 809, 818], "first_lost": [], "terminal_gained": [808, 809], "terminal_lost": []} |
| 81–180 | K25 V−，父 μ | B | 17／0 | 6／0 | {"none->none": 3, "right->front": 17} | {"first_gained": [800, 802, 804, 805, 806, 807, 808, 809, 810, 811, 812, 814, 815, 816, 817, 818, 819], "first_lost": [], "terminal_gained": [804, 805, 806, 811, 818, 819], "terminal_lost": []} |
| 81–180 | K25 V−，父 μ | joint | 3／0 | 0／0 | {} | {"first_gained": [806, 809, 818], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | 2.959 倍自身残差，晚期 μ | A | 0／0 | 0／0 | {"none->none": 3, "right->back": 8, "right->front": 2, "right->right": 7} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | 2.959 倍自身残差，晚期 μ | B | 0／0 | 0／0 | {"none->none": 3, "right->none": 10, "right->right": 7} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | 2.959 倍自身残差，晚期 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | 2.959 倍自身残差，父 μ | A | 10／0 | 10／0 | {"none->none": 3, "right->front": 7, "right->left": 10} | {"first_gained": [801, 802, 806, 807, 808, 809, 812, 814, 815, 818], "first_lost": [], "terminal_gained": [800, 801, 802, 806, 807, 808, 809, 814, 817, 818], "terminal_lost": []} |
| 81–180 | 2.959 倍自身残差，父 μ | B | 0／0 | 0／0 | {"none->none": 3, "right->none": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | 2.959 倍自身残差，父 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | 只换父 μ，父 μ | A | 18／0 | 17／0 | {"none->left": 2, "none->none": 1, "right->left": 16, "right->none": 1} | {"first_gained": [800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 817, 818, 819], "first_lost": [], "terminal_gained": [800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 811, 812, 813, 814, 817, 818, 819], "terminal_lost": []} |
| 1–400 | 只换父 μ，父 μ | B | 18／0 | 3／0 | {"none->front": 3, "right->front": 15, "right->left": 2} | {"first_gained": [801, 802, 803, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819], "first_lost": [], "terminal_gained": [802, 805, 809], "terminal_lost": []} |
| 1–400 | 只换父 μ，父 μ | joint | 16／0 | 3／0 | {} | {"first_gained": [801, 802, 803, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 817, 818, 819], "first_lost": [], "terminal_gained": [802, 805, 809], "terminal_lost": []} |
| 1–400 | K25 V＋，晚期 μ | A | 0／0 | 0／0 | {"none->none": 2, "none->right": 1, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | K25 V＋，晚期 μ | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | K25 V＋，晚期 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | K25 V＋，父 μ | A | 17／0 | 16／0 | {"none->left": 2, "none->none": 1, "right->left": 15, "right->none": 2} | {"first_gained": [800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 811, 812, 813, 814, 817, 818, 819], "first_lost": [], "terminal_gained": [801, 802, 803, 804, 805, 806, 807, 808, 809, 811, 812, 813, 814, 817, 818, 819], "terminal_lost": []} |
| 1–400 | K25 V＋，父 μ | B | 14／0 | 7／0 | {"none->front": 1, "none->none": 2, "right->front": 13, "right->left": 1, "right->none": 3} | {"first_gained": [802, 804, 805, 806, 807, 808, 809, 811, 813, 814, 815, 817, 818, 819], "first_lost": [], "terminal_gained": [802, 805, 806, 807, 808, 809, 811], "terminal_lost": []} |
| 1–400 | K25 V＋，父 μ | joint | 13／0 | 7／0 | {} | {"first_gained": [802, 804, 805, 806, 807, 808, 809, 811, 813, 814, 817, 818, 819], "first_lost": [], "terminal_gained": [802, 805, 806, 807, 808, 809, 811], "terminal_lost": []} |
| 1–400 | K25 V−，晚期 μ | A | 0／0 | 0／0 | {"none->none": 2, "none->right": 1, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | K25 V−，晚期 μ | B | 0／0 | 0／0 | {"none->none": 2, "none->right": 1, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | K25 V−，晚期 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | K25 V−，父 μ | A | 20／0 | 14／0 | {"none->left": 3, "right->left": 17} | {"first_gained": [800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819], "first_lost": [], "terminal_gained": [800, 803, 805, 806, 807, 808, 809, 811, 812, 813, 814, 817, 818, 819], "terminal_lost": []} |
| 1–400 | K25 V−，父 μ | B | 9／0 | 1／0 | {"none->front": 1, "none->left": 2, "right->front": 8, "right->left": 7, "right->none": 2} | {"first_gained": [802, 803, 805, 806, 807, 808, 809, 811, 816], "first_lost": [], "terminal_gained": [805], "terminal_lost": []} |
| 1–400 | K25 V−，父 μ | joint | 9／0 | 1／0 | {} | {"first_gained": [802, 803, 805, 806, 807, 808, 809, 811, 816], "first_lost": [], "terminal_gained": [805], "terminal_lost": []} |
| 1–400 | 2.959 倍自身残差，晚期 μ | A | 0／0 | 0／0 | {"none->front": 1, "none->none": 2, "right->back": 2, "right->front": 13, "right->right": 2} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | 2.959 倍自身残差，晚期 μ | B | 0／0 | 0／0 | {"none->none": 3, "right->none": 15, "right->right": 2} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | 2.959 倍自身残差，晚期 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | 2.959 倍自身残差，父 μ | A | 17／0 | 14／0 | {"none->left": 2, "none->none": 1, "right->left": 15, "right->none": 2} | {"first_gained": [800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 811, 812, 813, 814, 817, 818, 819], "first_lost": [], "terminal_gained": [800, 802, 803, 805, 806, 807, 808, 809, 812, 813, 814, 817, 818, 819], "terminal_lost": []} |
| 1–400 | 2.959 倍自身残差，父 μ | B | 3／0 | 0／0 | {"none->front": 1, "none->none": 2, "right->front": 2, "right->none": 15} | {"first_gained": [800, 811, 813], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | 2.959 倍自身残差，父 μ | joint | 3／0 | 0／0 | {} | {"first_gained": [800, 811, 813], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |

## 只改变 μ 来源

| 窗口 | 后条件 | 目标／joint | 首达获益／损失 | 终态获益／损失 | 前→后首达转移 | 获益／损失初态编号 |
| --- | --- | --- | --- | --- | --- | --- |
| 81–180 | K25 V＋，父 μ | A | 6／0 | 5／0 | {"none->none": 3, "right->front": 11, "right->left": 6} | {"first_gained": [801, 802, 806, 809, 814, 818], "first_lost": [], "terminal_gained": [807, 808, 809, 814, 817], "terminal_lost": []} |
| 81–180 | K25 V＋，父 μ | B | 12／0 | 11／0 | {"none->none": 3, "right->front": 12, "right->none": 5} | {"first_gained": [800, 804, 805, 807, 808, 809, 811, 812, 814, 817, 818, 819], "first_lost": [], "terminal_gained": [804, 805, 807, 808, 809, 811, 812, 814, 817, 818, 819], "terminal_lost": []} |
| 81–180 | K25 V＋，父 μ | joint | 3／0 | 5／0 | {} | {"first_gained": [809, 814, 818], "first_lost": [], "terminal_gained": [807, 808, 809, 814, 817], "terminal_lost": []} |
| 81–180 | K25 V−，父 μ | A | 4／0 | 2／0 | {"none->none": 3, "right->front": 13, "right->left": 4} | {"first_gained": [801, 806, 809, 818], "first_lost": [], "terminal_gained": [808, 809], "terminal_lost": []} |
| 81–180 | K25 V−，父 μ | B | 17／0 | 6／0 | {"none->none": 3, "right->front": 17} | {"first_gained": [800, 802, 804, 805, 806, 807, 808, 809, 810, 811, 812, 814, 815, 816, 817, 818, 819], "first_lost": [], "terminal_gained": [804, 805, 806, 811, 818, 819], "terminal_lost": []} |
| 81–180 | K25 V−，父 μ | joint | 3／0 | 0／0 | {} | {"first_gained": [806, 809, 818], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | 2.959 倍自身残差，父 μ | A | 10／0 | 10／0 | {"back->front": 2, "back->left": 6, "front->left": 2, "none->none": 3, "right->front": 5, "right->left": 2} | {"first_gained": [801, 802, 806, 807, 808, 809, 812, 814, 815, 818], "first_lost": [], "terminal_gained": [800, 801, 802, 806, 807, 808, 809, 814, 817, 818], "terminal_lost": []} |
| 81–180 | 2.959 倍自身残差，父 μ | B | 0／0 | 0／0 | {"none->none": 13, "right->none": 7} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | 2.959 倍自身残差，父 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | K25 V＋，父 μ | A | 17／0 | 16／0 | {"none->left": 1, "none->none": 1, "right->left": 16, "right->none": 2} | {"first_gained": [800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 811, 812, 813, 814, 817, 818, 819], "first_lost": [], "terminal_gained": [801, 802, 803, 804, 805, 806, 807, 808, 809, 811, 812, 813, 814, 817, 818, 819], "terminal_lost": []} |
| 1–400 | K25 V＋，父 μ | B | 14／0 | 7／0 | {"none->front": 1, "none->none": 2, "right->front": 13, "right->left": 1, "right->none": 3} | {"first_gained": [802, 804, 805, 806, 807, 808, 809, 811, 813, 814, 815, 817, 818, 819], "first_lost": [], "terminal_gained": [802, 805, 806, 807, 808, 809, 811], "terminal_lost": []} |
| 1–400 | K25 V＋，父 μ | joint | 13／0 | 7／0 | {} | {"first_gained": [802, 804, 805, 806, 807, 808, 809, 811, 813, 814, 817, 818, 819], "first_lost": [], "terminal_gained": [802, 805, 806, 807, 808, 809, 811], "terminal_lost": []} |
| 1–400 | K25 V−，父 μ | A | 20／0 | 14／0 | {"none->left": 2, "right->left": 18} | {"first_gained": [800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819], "first_lost": [], "terminal_gained": [800, 803, 805, 806, 807, 808, 809, 811, 812, 813, 814, 817, 818, 819], "terminal_lost": []} |
| 1–400 | K25 V−，父 μ | B | 9／0 | 1／0 | {"none->front": 1, "none->left": 1, "right->front": 8, "right->left": 8, "right->none": 2} | {"first_gained": [802, 803, 805, 806, 807, 808, 809, 811, 816], "first_lost": [], "terminal_gained": [805], "terminal_lost": []} |
| 1–400 | K25 V−，父 μ | joint | 9／0 | 1／0 | {} | {"first_gained": [802, 803, 805, 806, 807, 808, 809, 811, 816], "first_lost": [], "terminal_gained": [805], "terminal_lost": []} |
| 1–400 | 2.959 倍自身残差，父 μ | A | 17／0 | 14／0 | {"back->none": 2, "front->left": 14, "none->left": 1, "none->none": 1, "right->left": 2} | {"first_gained": [800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 811, 812, 813, 814, 817, 818, 819], "first_lost": [], "terminal_gained": [800, 802, 803, 805, 806, 807, 808, 809, 812, 813, 814, 817, 818, 819], "terminal_lost": []} |
| 1–400 | 2.959 倍自身残差，父 μ | B | 3／0 | 0／0 | {"none->front": 2, "none->none": 16, "right->front": 1, "right->none": 1} | {"first_gained": [800, 811, 813], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | 2.959 倍自身残差，父 μ | joint | 3／0 | 0／0 | {} | {"first_gained": [800, 811, 813], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |

## 在父 μ 上追加 V

| 窗口 | 后条件 | 目标／joint | 首达获益／损失 | 终态获益／损失 | 前→后首达转移 | 获益／损失初态编号 |
| --- | --- | --- | --- | --- | --- | --- |
| 81–180 | K25 V＋，父 μ | A | 1／0 | 2／4 | {"front->front": 11, "front->left": 1, "left->left": 5, "none->none": 3} | {"first_gained": [802], "first_lost": [], "terminal_gained": [814, 817], "terminal_lost": [801, 804, 806, 819]} |
| 81–180 | K25 V＋，父 μ | B | 0／3 | 1／2 | {"front->front": 12, "front->none": 3, "none->none": 5} | {"first_gained": [], "first_lost": [802, 806, 810], "terminal_gained": [817], "terminal_lost": [802, 810]} |
| 81–180 | K25 V＋，父 μ | joint | 0／1 | 2／2 | {} | {"first_gained": [], "first_lost": [806], "terminal_gained": [814, 817], "terminal_lost": [804, 819]} |
| 81–180 | K25 V−，父 μ | A | 0／1 | 0／5 | {"front->front": 12, "left->front": 1, "left->left": 4, "none->none": 3} | {"first_gained": [], "first_lost": [814], "terminal_gained": [], "terminal_lost": [801, 804, 806, 807, 819]} |
| 81–180 | K25 V−，父 μ | B | 2／0 | 1／7 | {"front->front": 15, "none->front": 2, "none->none": 3} | {"first_gained": [815, 816], "first_lost": [], "terminal_gained": [806], "terminal_lost": [802, 807, 808, 809, 810, 812, 814]} |
| 81–180 | K25 V−，父 μ | joint | 0／1 | 0／5 | {} | {"first_gained": [], "first_lost": [814], "terminal_gained": [], "terminal_lost": [804, 807, 808, 809, 819]} |
| 1–400 | K25 V＋，父 μ | A | 0／1 | 0／1 | {"left->left": 17, "left->none": 1, "none->none": 2} | {"first_gained": [], "first_lost": [810], "terminal_gained": [], "terminal_lost": [800]} |
| 1–400 | K25 V＋，父 μ | B | 1／5 | 4／0 | {"front->front": 13, "front->none": 5, "left->front": 1, "left->left": 1} | {"first_gained": [804], "first_lost": [801, 803, 810, 812, 816], "terminal_gained": [806, 807, 808, 811], "terminal_lost": []} |
| 1–400 | K25 V＋，父 μ | joint | 1／4 | 4／0 | {} | {"first_gained": [804], "first_lost": [801, 803, 810, 812], "terminal_gained": [806, 807, 808, 811], "terminal_lost": []} |
| 1–400 | K25 V−，父 μ | A | 2／0 | 0／3 | {"left->left": 18, "none->left": 2} | {"first_gained": [815, 816], "first_lost": [], "terminal_gained": [], "terminal_lost": [801, 802, 804]} |
| 1–400 | K25 V−，父 μ | B | 0／9 | 0／2 | {"front->front": 9, "front->left": 7, "front->none": 2, "left->left": 2} | {"first_gained": [], "first_lost": [801, 810, 812, 813, 814, 815, 817, 818, 819], "terminal_gained": [], "terminal_lost": [802, 809]} |
| 1–400 | K25 V−，父 μ | joint | 1／8 | 0／2 | {} | {"first_gained": [816], "first_lost": [801, 810, 812, 813, 814, 817, 818, 819], "terminal_gained": [], "terminal_lost": [802, 809]} |

## 只翻转 V 符号，相对正向

| 窗口 | 后条件 | 目标／joint | 首达获益／损失 | 终态获益／损失 | 前→后首达转移 | 获益／损失初态编号 |
| --- | --- | --- | --- | --- | --- | --- |
| 81–180 | K25 V−，晚期 μ | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | K25 V−，晚期 μ | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | K25 V−，晚期 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | K25 V−，父 μ | A | 0／2 | 0／3 | {"front->front": 11, "left->front": 2, "left->left": 4, "none->none": 3} | {"first_gained": [], "first_lost": [802, 814], "terminal_gained": [], "terminal_lost": [807, 814, 817]} |
| 81–180 | K25 V−，父 μ | B | 5／0 | 1／6 | {"front->front": 12, "none->front": 5, "none->none": 3} | {"first_gained": [802, 806, 810, 815, 816], "first_lost": [], "terminal_gained": [806], "terminal_lost": [807, 808, 809, 812, 814, 817]} |
| 81–180 | K25 V−，父 μ | joint | 1／1 | 0／5 | {} | {"first_gained": [806], "first_lost": [814], "terminal_gained": [], "terminal_lost": [807, 808, 809, 814, 817]} |
| 1–400 | K25 V−，晚期 μ | A | 0／0 | 0／0 | {"none->none": 1, "none->right": 1, "right->none": 1, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | K25 V−，晚期 μ | B | 0／0 | 0／0 | {"none->none": 2, "none->right": 1, "right->right": 17} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | K25 V−，晚期 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | K25 V−，父 μ | A | 3／0 | 1／3 | {"left->left": 17, "none->left": 3} | {"first_gained": [810, 815, 816], "first_lost": [], "terminal_gained": [800], "terminal_lost": [801, 802, 804]} |
| 1–400 | K25 V−，父 μ | B | 2／7 | 0／6 | {"front->front": 7, "front->left": 6, "front->none": 1, "left->left": 1, "none->front": 2, "none->left": 2, "none->none": 1} | {"first_gained": [803, 816], "first_lost": [804, 813, 814, 815, 817, 818, 819], "terminal_gained": [], "terminal_lost": [802, 806, 807, 808, 809, 811]} |
| 1–400 | K25 V−，父 μ | joint | 2／6 | 0／6 | {} | {"first_gained": [803, 816], "first_lost": [804, 813, 814, 817, 818, 819], "terminal_gained": [], "terminal_lost": [802, 806, 807, 808, 809, 811]} |

## 固定 μ，放大实际条件残差，相对 β=1

| 窗口 | 后条件 | 目标／joint | 首达获益／损失 | 终态获益／损失 | 前→后首达转移 | 获益／损失初态编号 |
| --- | --- | --- | --- | --- | --- | --- |
| 81–180 | 2.959 倍自身残差，晚期 μ | A | 0／0 | 0／0 | {"none->none": 3, "right->back": 8, "right->front": 2, "right->right": 7} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | 2.959 倍自身残差，晚期 μ | B | 0／0 | 0／0 | {"none->none": 3, "right->none": 10, "right->right": 7} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | 2.959 倍自身残差，晚期 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 81–180 | 2.959 倍自身残差，父 μ | A | 5／0 | 5／2 | {"front->front": 7, "front->left": 5, "left->left": 5, "none->none": 3} | {"first_gained": [802, 807, 808, 812, 815], "first_lost": [], "terminal_gained": [800, 802, 814, 817, 818], "terminal_lost": [804, 819]} |
| 81–180 | 2.959 倍自身残差，父 μ | B | 0／15 | 0／12 | {"front->none": 15, "none->none": 5} | {"first_gained": [], "first_lost": [800, 802, 804, 805, 806, 807, 808, 809, 810, 811, 812, 814, 817, 818, 819], "terminal_gained": [], "terminal_lost": [802, 804, 805, 807, 808, 809, 810, 811, 812, 814, 818, 819]} |
| 81–180 | 2.959 倍自身残差，父 μ | joint | 0／4 | 0／5 | {} | {"first_gained": [], "first_lost": [806, 809, 814, 818], "terminal_gained": [], "terminal_lost": [804, 807, 808, 809, 819]} |
| 1–400 | 2.959 倍自身残差，晚期 μ | A | 0／0 | 0／0 | {"none->front": 1, "none->none": 2, "right->back": 2, "right->front": 13, "right->right": 2} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | 2.959 倍自身残差，晚期 μ | B | 0／0 | 0／0 | {"none->none": 3, "right->none": 15, "right->right": 2} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | 2.959 倍自身残差，晚期 μ | joint | 0／0 | 0／0 | {} | {"first_gained": [], "first_lost": [], "terminal_gained": [], "terminal_lost": []} |
| 1–400 | 2.959 倍自身残差，父 μ | A | 0／1 | 0／3 | {"left->left": 17, "left->none": 1, "none->none": 2} | {"first_gained": [], "first_lost": [810], "terminal_gained": [], "terminal_lost": [801, 804, 811]} |
| 1–400 | 2.959 倍自身残差，父 μ | B | 1／16 | 0／3 | {"front->front": 2, "front->none": 16, "left->front": 1, "left->none": 1} | {"first_gained": [800], "first_lost": [801, 802, 803, 805, 806, 807, 808, 809, 810, 812, 814, 815, 816, 817, 818, 819], "terminal_gained": [], "terminal_lost": [802, 805, 809]} |
| 1–400 | 2.959 倍自身残差，父 μ | joint | 1／14 | 0／3 | {} | {"first_gained": [800], "first_lost": [801, 802, 803, 805, 806, 807, 808, 809, 810, 812, 814, 817, 818, 819], "terminal_gained": [], "terminal_lost": [802, 805, 809]} |
