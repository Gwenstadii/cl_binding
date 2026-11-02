# K3-E2b（事后补充）：保留物体名词的文本控制

回合 180/180。事后补充块，不与 E2 合并判定。

| 家族 | 模型 | 指令 | 文本 | 捕获 | 任何目标 | 抬起 | 首次目标分布 |
|---|---|---|---|---:|---:|---:|---|
| scene2 | R0 | W1 | there is a book and a caddy on the desk | 0.0 | 0.4 | 0.6 | {'left': 3, 'none': 6, 'front': 1} |
| scene2 | R0 | W2 | pick up the book | 0.0 | 0.3 | 0.8 | {'left': 2, 'none': 7, 'front': 1} |
| scene2 | R0 | W3 | put the book in the caddy | 0.0 | 0.6 | 0.8 | {'left': 3, 'none': 4, 'front': 3} |
| scene2 | FT4 | W1 | there is a book and a caddy on the desk | 0.5 | 0.5 | 0.8 | {'none': 5, 'right': 5} |
| scene2 | FT4 | W2 | pick up the book | 0.5 | 0.5 | 0.7 | {'none': 5, 'right': 5} |
| scene2 | FT4 | W3 | put the book in the caddy | 0.8 | 0.8 | 0.9 | {'right': 8, 'none': 2} |
| scene1b | R0 | W1 | there is a book and a caddy on the desk | 0.0 | 0.2 | 0.3 | {'none': 8, 'front': 2} |
| scene1b | R0 | W2 | pick up the book | 0.0 | 0.0 | 0.2 | {'none': 10} |
| scene1b | R0 | W3 | put the book in the caddy | 0.0 | 0.2 | 0.3 | {'none': 8, 'front': 2} |
| scene1b | FT4 | W1 | there is a book and a caddy on the desk | 0.0 | 0.0 | 0.0 | {'none': 10} |
| scene1b | FT4 | W2 | pick up the book | 0.0 | 0.0 | 0.0 | {'none': 10} |
| scene1b | FT4 | W3 | put the book in the caddy | 0.1 | 0.1 | 0.1 | {'none': 9, 'right': 1} |
| goal | R0 | W1 | there is a bowl and a plate on the table | 0.0 | 0.4 | 0.7 | {'none': 6, 'front': 4} |
| goal | R0 | W2 | pick up the bowl | 0.0 | 0.8 | 0.9 | {'none': 2, 'left': 8} |
| goal | R0 | W3 | put the bowl on the table | 0.0 | 0.8 | 0.9 | {'none': 2, 'left': 8} |
| goal | FT4 | W1 | there is a bowl and a plate on the table | 0.9 | 0.9 | 0.9 | {'none': 1, 'right': 9} |
| goal | FT4 | W2 | pick up the bowl | 1.0 | 1.0 | 1.0 | {'right': 10} |
| goal | FT4 | W3 | put the bowl on the table | 0.9 | 0.9 | 1.0 | {'right': 9, 'none': 1} |
