# nlp_transformers_tasks

本项目主要是基于[transformers](https://github.com/huggingface/transformers)来实现一些常见的nlp任务。使用者根据需要可以找到对应的任务，将代码中的数据替换就可以训练自己的模型。

目前已经实现的NLP 任务如下（更新中）：

### 1.[文本分类(Text Classification)](https://github.com/littletrain-jyp/nlp_transformers_tasks/tree/main/text_classification)
> 对给定文本进行分类，常用于`情感分析`，`意图识别`，`新闻分类`等
- 基于bert等预训练模型的文本分类，支持单标签分类与多标签分类等

### 2.命名实体识别
> 识别给定文本中的实体，如识别出一句话中的股票代码、人名等。