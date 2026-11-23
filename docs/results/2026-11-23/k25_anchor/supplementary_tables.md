# K25 锚点审计：完整补充表

所有成功数的分母为 20；获益／损失按同一初态配对。A／B 的实际目标在每节注明。没有将不同目标的最优配置拼接成策略。

## Left＋Right→Front：A=Left，B=Right，N=Front

### 完整供体

| 条件 | 目标 | 首达正确 | 终态正确 | 曾放好 | 抓取 | 首达分布 |
| --- | --- | --- | --- | --- | --- | --- |
| 父模型0 | A | 20 | 20 | 20 | 20 | {"left": 20} |
| 父模型0 | B | 18 | 17 | 17 | 18 | {"none": 2, "right": 18} |
| 父模型0 | N | 17 | 3 | 3 | 18 | {"front": 17, "none": 3} |
| 父模型30 | A | 19 | 16 | 19 | 19 | {"left": 19, "none": 1} |
| 父模型30 | B | 18 | 11 | 11 | 18 | {"none": 2, "right": 18} |
| 父模型30 | N | 18 | 4 | 5 | 18 | {"front": 18, "none": 2} |
| 晚期1000 | A | 17 | 12 | 17 | 20 | {"left": 17, "none": 3} |
| 晚期1000 | B | 3 | 0 | 0 | 20 | {"front": 8, "none": 9, "right": 3} |
| 晚期1000 | N | 18 | 15 | 17 | 20 | {"front": 18, "none": 2} |

### 只换八张 V

| 条件 | 目标 | 首达正确 | 终态正确 | 曾放好 | 抓取 | 首达分布 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | A | 19 | 6 | 20 | 20 | {"left": 19, "none": 1} |
| V0 | B | 8 | 1 | 1 | 20 | {"front": 5, "none": 7, "right": 8} |
| V30 | A | 19 | 5 | 20 | 20 | {"left": 19, "none": 1} |
| V30 | B | 11 | 3 | 3 | 20 | {"front": 3, "none": 6, "right": 11} |

### 全部区别修正及读出

| 锚点 / α / 窗口 / 符号 | 目标 | 首达正确 | 终态正确 | 曾放好 | 抓取 | 首达分布 | 首次读出 | 全块读对回合数 | 并列块数 | 首块 Δd RMS 中位数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 / 1 / 81–180 / forward | A | 17 | 11 | 18 | 20 | {"left": 17, "none": 3} | {"A": 20} | 20 | 0 | 0.057910628 |
| 0 / 1 / 81–180 / forward | B | 6 | 1 | 1 | 20 | {"front": 2, "none": 12, "right": 6} | {"B": 20} | 20 | 0 | 0.061313888 |
| 0 / 1 / 81–180 / reverse | A | 17 | 13 | 17 | 20 | {"left": 17, "none": 3} | {"A": 20} | 20 | 0 | 0.058635078 |
| 0 / 1 / 81–180 / reverse | B | 0 | 0 | 0 | 20 | {"front": 9, "none": 11} | {"B": 20} | 20 | 0 | 0.061271001 |
| 0 / 1 / 1–400 / forward | A | 17 | 10 | 17 | 20 | {"left": 17, "none": 3} | {"A": 20} | 20 | 0 | 0.034006995 |
| 0 / 1 / 1–400 / forward | B | 9 | 1 | 1 | 20 | {"back": 1, "front": 1, "none": 9, "right": 9} | {"B": 20} | 20 | 0 | 0.034815524 |
| 0 / 1 / 1–400 / reverse | A | 17 | 14 | 17 | 20 | {"left": 17, "none": 3} | {"A": 20} | 20 | 0 | 0.033722602 |
| 0 / 1 / 1–400 / reverse | B | 0 | 0 | 0 | 20 | {"front": 9, "none": 11} | {"B": 20} | 20 | 0 | 0.03606185 |
| 0 / 5.29404 / 81–180 / forward | A | 14 | 9 | 15 | 20 | {"left": 14, "none": 6} | {"A": 20} | 20 | 0 | 0.055869203 |
| 0 / 5.29404 / 81–180 / forward | B | 15 | 5 | 5 | 20 | {"none": 5, "right": 15} | {"B": 20} | 20 | 0 | 0.060209245 |
| 0 / 5.29404 / 81–180 / reverse | A | 17 | 14 | 18 | 20 | {"front": 1, "left": 17, "none": 2} | {"A": 20} | 20 | 0 | 0.058382972 |
| 0 / 5.29404 / 81–180 / reverse | B | 0 | 0 | 0 | 20 | {"front": 14, "none": 6} | {"B": 20} | 20 | 0 | 0.060889949 |
| 0 / 5.29404 / 1–400 / forward | A | 13 | 8 | 13 | 20 | {"left": 13, "none": 7} | {"A": 20} | 20 | 0 | 0.035000645 |
| 0 / 5.29404 / 1–400 / forward | B | 11 | 3 | 3 | 19 | {"none": 9, "right": 11} | {"B": 20} | 20 | 0 | 0.033281493 |
| 0 / 5.29404 / 1–400 / reverse | A | 16 | 13 | 16 | 20 | {"front": 2, "left": 16, "none": 2} | {"A": 20} | 20 | 0 | 0.033851617 |
| 0 / 5.29404 / 1–400 / reverse | B | 0 | 0 | 0 | 20 | {"front": 8, "none": 12} | {"B": 20} | 20 | 0 | 0.044295036 |
| 30 / 1 / 81–180 / forward | A | 16 | 9 | 17 | 20 | {"left": 16, "none": 4} | {"A": 20} | 20 | 0 | 0.045709443 |
| 30 / 1 / 81–180 / forward | B | 7 | 0 | 0 | 20 | {"back": 1, "front": 3, "none": 9, "right": 7} | {"B": 20} | 20 | 0 | 0.043530906 |
| 30 / 1 / 81–180 / reverse | A | 17 | 13 | 17 | 20 | {"left": 17, "none": 3} | {"A": 20} | 20 | 0 | 0.045770526 |
| 30 / 1 / 81–180 / reverse | B | 1 | 0 | 0 | 20 | {"back": 1, "front": 7, "none": 11, "right": 1} | {"B": 20} | 20 | 0 | 0.043715835 |
| 30 / 1 / 1–400 / forward | A | 18 | 10 | 18 | 20 | {"left": 18, "none": 2} | {"A": 20} | 20 | 0 | 0.029126705 |
| 30 / 1 / 1–400 / forward | B | 6 | 1 | 1 | 20 | {"front": 3, "none": 11, "right": 6} | {"B": 20} | 20 | 0 | 0.030269086 |
| 30 / 1 / 1–400 / reverse | A | 17 | 11 | 17 | 20 | {"left": 17, "none": 3} | {"A": 20} | 20 | 0 | 0.029302717 |
| 30 / 1 / 1–400 / reverse | B | 0 | 0 | 0 | 20 | {"front": 12, "none": 8} | {"B": 20} | 20 | 0 | 0.030863462 |
| 30 / 5.29404 / 81–180 / forward | A | 16 | 9 | 16 | 20 | {"left": 16, "none": 4} | {"A": 20} | 20 | 0 | 0.045289967 |
| 30 / 5.29404 / 81–180 / forward | B | 14 | 7 | 7 | 20 | {"none": 6, "right": 14} | {"B": 20} | 20 | 0 | 0.043464106 |
| 30 / 5.29404 / 81–180 / reverse | A | 18 | 13 | 18 | 20 | {"left": 18, "none": 2} | {"A": 20} | 20 | 0 | 0.046187835 |
| 30 / 5.29404 / 81–180 / reverse | B | 0 | 0 | 0 | 20 | {"front": 11, "none": 9} | {"B": 20} | 20 | 0 | 0.042999274 |
| 30 / 5.29404 / 1–400 / forward | A | 15 | 9 | 15 | 20 | {"left": 15, "none": 5} | {"A": 20} | 20 | 0 | 0.029090157 |
| 30 / 5.29404 / 1–400 / forward | B | 11 | 5 | 5 | 19 | {"none": 9, "right": 11} | {"B": 20} | 20 | 0 | 0.029396365 |
| 30 / 5.29404 / 1–400 / reverse | A | 19 | 13 | 19 | 20 | {"left": 19, "none": 1} | {"A": 20} | 20 | 0 | 0.028688056 |
| 30 / 5.29404 / 1–400 / reverse | B | 0 | 0 | 0 | 20 | {"front": 7, "none": 13} | {"B": 20} | 20 | 0 | 0.037264491 |

首块 Δd RMS 为动作速度场七个执行维度、十次去噪的均值，再取 20 回合中位数；它不是物理位移或最终放置分数。全块读对以该条件实际激活的所有修正块为准。

### 相对晚期原生的逐初态获益／损失

| 配置 | 目标 | 首达获益／损失 | 终态获益／损失 | 首达转移 |
| --- | --- | --- | --- | --- |
| 0 / 1 / 81–180 / forward | A | 0／0 | 1／2 | {"left->left": 17, "none->none": 3} |
| 0 / 1 / 81–180 / forward | B | 3／0 | 1／0 | {"front->front": 2, "front->none": 4, "front->right": 2, "none->none": 8, "none->right": 1, "right->right": 3} |
| 0 / 1 / 81–180 / reverse | A | 0／0 | 2／1 | {"left->left": 17, "none->none": 3} |
| 0 / 1 / 81–180 / reverse | B | 0／3 | 0／0 | {"front->front": 8, "none->none": 9, "right->front": 1, "right->none": 2} |
| 0 / 1 / 1–400 / forward | A | 1／1 | 1／3 | {"left->left": 16, "left->none": 1, "none->left": 1, "none->none": 2} |
| 0 / 1 / 1–400 / forward | B | 6／0 | 1／0 | {"front->back": 1, "front->front": 1, "front->none": 1, "front->right": 5, "none->none": 8, "none->right": 1, "right->right": 3} |
| 0 / 1 / 1–400 / reverse | A | 1／1 | 4／2 | {"left->left": 16, "left->none": 1, "none->left": 1, "none->none": 2} |
| 0 / 1 / 1–400 / reverse | B | 0／3 | 0／0 | {"front->front": 6, "front->none": 2, "none->front": 1, "none->none": 8, "right->front": 2, "right->none": 1} |
| 0 / 5.29404 / 81–180 / forward | A | 0／3 | 1／4 | {"left->left": 14, "left->none": 3, "none->none": 3} |
| 0 / 5.29404 / 81–180 / forward | B | 12／0 | 5／0 | {"front->none": 2, "front->right": 6, "none->none": 3, "none->right": 6, "right->right": 3} |
| 0 / 5.29404 / 81–180 / reverse | A | 2／2 | 3／1 | {"left->front": 1, "left->left": 15, "left->none": 1, "none->left": 2, "none->none": 1} |
| 0 / 5.29404 / 81–180 / reverse | B | 0／3 | 0／0 | {"front->front": 8, "none->front": 4, "none->none": 5, "right->front": 2, "right->none": 1} |
| 0 / 5.29404 / 1–400 / forward | A | 1／5 | 3／7 | {"left->left": 12, "left->none": 5, "none->left": 1, "none->none": 2} |
| 0 / 5.29404 / 1–400 / forward | B | 8／0 | 3／0 | {"front->none": 4, "front->right": 4, "none->none": 5, "none->right": 4, "right->right": 3} |
| 0 / 5.29404 / 1–400 / reverse | A | 3／4 | 4／3 | {"left->front": 2, "left->left": 13, "left->none": 2, "none->left": 3} |
| 0 / 5.29404 / 1–400 / reverse | B | 0／3 | 0／0 | {"front->front": 5, "front->none": 3, "none->front": 3, "none->none": 6, "right->none": 3} |
| 30 / 1 / 81–180 / forward | A | 0／1 | 1／4 | {"left->left": 16, "left->none": 1, "none->none": 3} |
| 30 / 1 / 81–180 / forward | B | 4／0 | 0／0 | {"front->back": 1, "front->front": 3, "front->none": 1, "front->right": 3, "none->none": 8, "none->right": 1, "right->right": 3} |
| 30 / 1 / 81–180 / reverse | A | 0／0 | 2／1 | {"left->left": 17, "none->none": 3} |
| 30 / 1 / 81–180 / reverse | B | 0／2 | 0／0 | {"front->front": 7, "front->none": 1, "none->none": 9, "right->back": 1, "right->none": 1, "right->right": 1} |
| 30 / 1 / 1–400 / forward | A | 2／1 | 2／4 | {"left->left": 16, "left->none": 1, "none->left": 2, "none->none": 1} |
| 30 / 1 / 1–400 / forward | B | 4／1 | 1／0 | {"front->front": 3, "front->none": 2, "front->right": 3, "none->none": 8, "none->right": 1, "right->none": 1, "right->right": 2} |
| 30 / 1 / 1–400 / reverse | A | 0／0 | 2／3 | {"left->left": 17, "none->none": 3} |
| 30 / 1 / 1–400 / reverse | B | 0／3 | 0／0 | {"front->front": 7, "front->none": 1, "none->front": 2, "none->none": 7, "right->front": 3} |
| 30 / 5.29404 / 81–180 / forward | A | 0／1 | 2／5 | {"left->left": 16, "left->none": 1, "none->none": 3} |
| 30 / 5.29404 / 81–180 / forward | B | 11／0 | 7／0 | {"front->none": 3, "front->right": 5, "none->none": 3, "none->right": 6, "right->right": 3} |
| 30 / 5.29404 / 81–180 / reverse | A | 2／1 | 3／2 | {"left->left": 16, "left->none": 1, "none->left": 2, "none->none": 1} |
| 30 / 5.29404 / 81–180 / reverse | B | 0／3 | 0／0 | {"front->front": 7, "front->none": 1, "none->front": 3, "none->none": 6, "right->front": 1, "right->none": 2} |
| 30 / 5.29404 / 1–400 / forward | A | 1／3 | 3／6 | {"left->left": 14, "left->none": 3, "none->left": 1, "none->none": 2} |
| 30 / 5.29404 / 1–400 / forward | B | 8／0 | 5／0 | {"front->none": 3, "front->right": 5, "none->none": 6, "none->right": 3, "right->right": 3} |
| 30 / 5.29404 / 1–400 / reverse | A | 3／1 | 5／4 | {"left->left": 16, "left->none": 1, "none->left": 3} |
| 30 / 5.29404 / 1–400 / reverse | B | 0／3 | 0／0 | {"front->front": 6, "front->none": 2, "none->front": 1, "none->none": 8, "right->none": 3} |

### 保持首次读出相同，翻转符号后的逐初态转移

| 反向配置（相对同配置正向） | 目标 | 首达获益／损失 | 终态获益／损失 | 正向→反向首达 |
| --- | --- | --- | --- | --- |
| 0 / 1 / 81–180 / reverse | A | 0／0 | 2／0 | {"left->left": 17, "none->none": 3} |
| 0 / 1 / 81–180 / reverse | B | 0／6 | 0／1 | {"front->front": 2, "none->front": 4, "none->none": 8, "right->front": 3, "right->none": 3} |
| 0 / 1 / 1–400 / reverse | A | 1／1 | 5／1 | {"left->left": 16, "left->none": 1, "none->left": 1, "none->none": 2} |
| 0 / 1 / 1–400 / reverse | B | 0／9 | 0／1 | {"back->front": 1, "front->none": 1, "none->front": 2, "none->none": 7, "right->front": 6, "right->none": 3} |
| 0 / 5.29404 / 81–180 / reverse | A | 5／2 | 6／1 | {"left->front": 1, "left->left": 12, "left->none": 1, "none->left": 5, "none->none": 1} |
| 0 / 5.29404 / 81–180 / reverse | B | 0／15 | 0／5 | {"none->front": 3, "none->none": 2, "right->front": 11, "right->none": 4} |
| 0 / 5.29404 / 1–400 / reverse | A | 5／2 | 5／0 | {"left->front": 1, "left->left": 11, "left->none": 1, "none->front": 1, "none->left": 5, "none->none": 1} |
| 0 / 5.29404 / 1–400 / reverse | B | 0／11 | 0／3 | {"none->front": 3, "none->none": 6, "right->front": 5, "right->none": 6} |
| 30 / 1 / 81–180 / reverse | A | 1／0 | 5／1 | {"left->left": 16, "none->left": 1, "none->none": 3} |
| 30 / 1 / 81–180 / reverse | B | 0／6 | 0／0 | {"back->front": 1, "front->front": 3, "none->front": 1, "none->none": 8, "right->back": 1, "right->front": 2, "right->none": 3, "right->right": 1} |
| 30 / 1 / 1–400 / reverse | A | 1／2 | 3／2 | {"left->left": 16, "left->none": 2, "none->left": 1, "none->none": 1} |
| 30 / 1 / 1–400 / reverse | B | 0／6 | 0／1 | {"front->front": 3, "none->front": 5, "none->none": 6, "right->front": 4, "right->none": 2} |
| 30 / 5.29404 / 81–180 / reverse | A | 3／1 | 6／2 | {"left->left": 15, "left->none": 1, "none->left": 3, "none->none": 1} |
| 30 / 5.29404 / 81–180 / reverse | B | 0／14 | 0／7 | {"none->front": 4, "none->none": 2, "right->front": 7, "right->none": 7} |
| 30 / 5.29404 / 1–400 / reverse | A | 5／1 | 8／4 | {"left->left": 14, "left->none": 1, "none->left": 5} |
| 30 / 5.29404 / 1–400 / reverse | B | 0／11 | 0／5 | {"none->front": 2, "none->none": 7, "right->front": 5, "right->none": 6} |

### 完整第 30 步相对父模型第 0 步

| 目标 | 首达获益／损失 | 终态获益／损失 | 0→30 首达转移 |
| --- | --- | --- | --- |
| A | 0／1 | 0／4 | {"left->left": 19, "left->none": 1} |
| B | 0／0 | 0／6 | {"none->none": 2, "right->right": 18} |
| N | 1／0 | 3／2 | {"front->front": 17, "none->front": 1, "none->none": 2} |

## Left＋Front→Right：A=Left，B=Front，N=Right

### 完整供体

| 条件 | 目标 | 首达正确 | 终态正确 | 曾放好 | 抓取 | 首达分布 |
| --- | --- | --- | --- | --- | --- | --- |
| 父模型0 | A | 17 | 16 | 17 | 17 | {"left": 17, "none": 3} |
| 父模型0 | B | 20 | 14 | 15 | 20 | {"front": 20} |
| 父模型0 | N | 0 | 0 | 0 | 20 | {"left": 20} |
| 父模型30 | A | 17 | 6 | 18 | 18 | {"front": 1, "left": 17, "none": 2} |
| 父模型30 | B | 9 | 2 | 4 | 18 | {"front": 9, "none": 11} |
| 父模型30 | N | 0 | 0 | 0 | 18 | {"front": 17, "none": 3} |
| 晚期1000 | A | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} |
| 晚期1000 | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} |
| 晚期1000 | N | 17 | 13 | 14 | 18 | {"none": 3, "right": 17} |

### 只换八张 V

| 条件 | 目标 | 首达正确 | 终态正确 | 曾放好 | 抓取 | 首达分布 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | A | 0 | 0 | 0 | 18 | {"back": 2, "front": 11, "none": 2, "right": 5} |
| V0 | B | 0 | 0 | 0 | 17 | {"none": 4, "right": 16} |
| V30 | A | 0 | 0 | 0 | 18 | {"front": 1, "none": 2, "right": 17} |
| V30 | B | 0 | 0 | 0 | 16 | {"none": 5, "right": 15} |

### 全部区别修正及读出

| 锚点 / α / 窗口 / 符号 | 目标 | 首达正确 | 终态正确 | 曾放好 | 抓取 | 首达分布 | 首次读出 | 全块读对回合数 | 并列块数 | 首块 Δd RMS 中位数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 / 1 / 81–180 / forward | A | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0.048131952 |
| 0 / 1 / 81–180 / forward | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.042403936 |
| 0 / 1 / 81–180 / reverse | A | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0.048340075 |
| 0 / 1 / 81–180 / reverse | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.041942342 |
| 0 / 1 / 1–400 / forward | A | 0 | 0 | 0 | 18 | {"none": 2, "right": 18} | {"A": 20} | 20 | 0 | 0.02559976 |
| 0 / 1 / 1–400 / forward | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.025184308 |
| 0 / 1 / 1–400 / reverse | A | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0.025603108 |
| 0 / 1 / 1–400 / reverse | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.025753493 |
| 0 / 5.29404 / 81–180 / forward | A | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0.04770592 |
| 0 / 5.29404 / 81–180 / forward | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.042445562 |
| 0 / 5.29404 / 81–180 / reverse | A | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0.047899324 |
| 0 / 5.29404 / 81–180 / reverse | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.042455082 |
| 0 / 5.29404 / 1–400 / forward | A | 0 | 0 | 0 | 19 | {"back": 2, "none": 1, "right": 17} | {"A": 20} | 20 | 0 | 0.026203241 |
| 0 / 5.29404 / 1–400 / forward | B | 0 | 0 | 0 | 14 | {"none": 7, "right": 13} | {"B": 20} | 20 | 0 | 0.024598272 |
| 0 / 5.29404 / 1–400 / reverse | A | 0 | 0 | 0 | 18 | {"none": 2, "right": 18} | {"A": 20} | 20 | 0 | 0.024752953 |
| 0 / 5.29404 / 1–400 / reverse | B | 0 | 0 | 0 | 18 | {"none": 2, "right": 18} | {"B": 20} | 20 | 0 | 0.026255189 |
| 30 / 1 / 81–180 / forward | A | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0.041783902 |
| 30 / 1 / 81–180 / forward | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.03447935 |
| 30 / 1 / 81–180 / reverse | A | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0.041525828 |
| 30 / 1 / 81–180 / reverse | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.034805558 |
| 30 / 1 / 1–400 / forward | A | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0.021983815 |
| 30 / 1 / 1–400 / forward | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.021674162 |
| 30 / 1 / 1–400 / reverse | A | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0.021496217 |
| 30 / 1 / 1–400 / reverse | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.021501342 |
| 30 / 5.29404 / 81–180 / forward | A | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0.042095351 |
| 30 / 5.29404 / 81–180 / forward | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.034475832 |
| 30 / 5.29404 / 81–180 / reverse | A | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"A": 20} | 20 | 0 | 0.040734452 |
| 30 / 5.29404 / 81–180 / reverse | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.035089041 |
| 30 / 5.29404 / 1–400 / forward | A | 0 | 0 | 0 | 18 | {"none": 2, "right": 18} | {"A": 20} | 20 | 0 | 0.022098172 |
| 30 / 5.29404 / 1–400 / forward | B | 0 | 0 | 0 | 17 | {"none": 3, "right": 17} | {"B": 20} | 20 | 0 | 0.020791956 |
| 30 / 5.29404 / 1–400 / reverse | A | 0 | 0 | 0 | 18 | {"none": 2, "right": 18} | {"A": 20} | 20 | 0 | 0.021156457 |
| 30 / 5.29404 / 1–400 / reverse | B | 0 | 0 | 0 | 18 | {"none": 2, "right": 18} | {"B": 20} | 20 | 0 | 0.021408275 |

首块 Δd RMS 为动作速度场七个执行维度、十次去噪的均值，再取 20 回合中位数；它不是物理位移或最终放置分数。全块读对以该条件实际激活的所有修正块为准。

### 相对晚期原生的逐初态获益／损失

| 配置 | 目标 | 首达获益／损失 | 终态获益／损失 | 首达转移 |
| --- | --- | --- | --- | --- |
| 0 / 1 / 81–180 / forward | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 1 / 81–180 / forward | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 1 / 81–180 / reverse | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 1 / 81–180 / reverse | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 1 / 1–400 / forward | A | 0／0 | 0／0 | {"none->none": 2, "none->right": 1, "right->right": 17} |
| 0 / 1 / 1–400 / forward | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 1 / 1–400 / reverse | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 1 / 1–400 / reverse | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 5.29404 / 81–180 / forward | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 5.29404 / 81–180 / forward | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 5.29404 / 81–180 / reverse | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 5.29404 / 81–180 / reverse | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 5.29404 / 1–400 / forward | A | 0／0 | 0／0 | {"none->back": 1, "none->none": 1, "none->right": 1, "right->back": 1, "right->right": 16} |
| 0 / 5.29404 / 1–400 / forward | B | 0／0 | 0／0 | {"none->none": 3, "right->none": 4, "right->right": 13} |
| 0 / 5.29404 / 1–400 / reverse | A | 0／0 | 0／0 | {"none->none": 2, "none->right": 1, "right->right": 17} |
| 0 / 5.29404 / 1–400 / reverse | B | 0／0 | 0／0 | {"none->none": 2, "none->right": 1, "right->right": 17} |
| 30 / 1 / 81–180 / forward | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 1 / 81–180 / forward | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 1 / 81–180 / reverse | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 1 / 81–180 / reverse | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 1 / 1–400 / forward | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 1 / 1–400 / forward | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 1 / 1–400 / reverse | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 1 / 1–400 / reverse | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 5.29404 / 81–180 / forward | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 5.29404 / 81–180 / forward | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 5.29404 / 81–180 / reverse | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 5.29404 / 81–180 / reverse | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 5.29404 / 1–400 / forward | A | 0／0 | 0／0 | {"none->none": 2, "none->right": 1, "right->right": 17} |
| 30 / 5.29404 / 1–400 / forward | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 5.29404 / 1–400 / reverse | A | 0／0 | 0／0 | {"none->none": 2, "none->right": 1, "right->right": 17} |
| 30 / 5.29404 / 1–400 / reverse | B | 0／0 | 0／0 | {"none->none": 2, "none->right": 1, "right->right": 17} |

### 保持首次读出相同，翻转符号后的逐初态转移

| 反向配置（相对同配置正向） | 目标 | 首达获益／损失 | 终态获益／损失 | 正向→反向首达 |
| --- | --- | --- | --- | --- |
| 0 / 1 / 81–180 / reverse | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 1 / 81–180 / reverse | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 1 / 1–400 / reverse | A | 0／0 | 0／0 | {"none->none": 2, "right->none": 1, "right->right": 17} |
| 0 / 1 / 1–400 / reverse | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 5.29404 / 81–180 / reverse | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 5.29404 / 81–180 / reverse | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 0 / 5.29404 / 1–400 / reverse | A | 0／0 | 0／0 | {"back->none": 1, "back->right": 1, "none->right": 1, "right->none": 1, "right->right": 16} |
| 0 / 5.29404 / 1–400 / reverse | B | 0／0 | 0／0 | {"none->none": 2, "none->right": 5, "right->right": 13} |
| 30 / 1 / 81–180 / reverse | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 1 / 81–180 / reverse | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 1 / 1–400 / reverse | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 1 / 1–400 / reverse | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 5.29404 / 81–180 / reverse | A | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 5.29404 / 81–180 / reverse | B | 0／0 | 0／0 | {"none->none": 3, "right->right": 17} |
| 30 / 5.29404 / 1–400 / reverse | A | 0／0 | 0／0 | {"none->none": 1, "none->right": 1, "right->none": 1, "right->right": 17} |
| 30 / 5.29404 / 1–400 / reverse | B | 0／0 | 0／0 | {"none->none": 2, "none->right": 1, "right->right": 17} |

### 完整第 30 步相对父模型第 0 步

| 目标 | 首达获益／损失 | 终态获益／损失 | 0→30 首达转移 |
| --- | --- | --- | --- |
| A | 2／2 | 0／10 | {"left->left": 15, "left->none": 2, "none->front": 1, "none->left": 2} |
| B | 0／11 | 1／13 | {"front->front": 9, "front->none": 11} |
| N | 0／0 | 0／0 | {"left->front": 17, "left->none": 3} |
