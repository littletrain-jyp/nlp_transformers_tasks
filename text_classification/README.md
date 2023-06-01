文本分类任务


针对文本分类的常见改进措施：
1.类别不平衡：focal_loss以及scale loss

# cnews
## hfl/chinese-bert-wwm-ext

统一运行400步停止,dev的结果
| model                      | acc    | micro_f1 | micro_f1 |
|----------------------------|--------|----------|-----------|
| baseline                  | 0.9597 | 0.9597   | 0.959667 |
| baseline+对抗                | 0.953  | 0.953    | 0.952861 |
| baseline+adamw+warmup        | 0.9527  | 0.9527 | 0.952820 |
| baseline+adamw+warmup+梯度累加| 0.9693  | 0.9693    | 0.969227  |
| baseline+adamw+warmup+梯度累加+autocast|0.9683  | 0.9683    | 0.968111   |

with_amp: 11.06min
without_amp: 14:05min

# 已完善的功能：

