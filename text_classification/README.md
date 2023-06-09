# 基于预训练模型的中文文本分类任务
本项目基于pytorch以及transformer库的中文文本分类。本项目基于[该项目](https://github.com/taishan1994/pytorch_bert_chinese_text_classification)修改而来。

## 相关说明
```
-- conf: 用于存放各种任务的conf文件，用于指定模型名称、数据集地址、学习率等配置。
-- data: 用于存放各个数据集
-- dataset: 用于存放读取各个数据集的类，在dataset.py文件中需要根据自己数据集的格式自定义对应的处理类。
-- models: 用于存放模型
-- utils: 用于存放一些辅助函数
-- classifiaction_trainer.py：是自己手写的trainer, 定义训练,预测,评估等流程
-- main.py： 主函数
-- run_train.sh: 训练脚本
```
- **配置文件**：把一个任务的文件保存成了一个yml文件
- **模型架构**：我这里直接使用了transformers自带的AutoModelForSequenceClassification，并没有使用特殊的模型结构，后续可根据具体的任务来定义自己的结构。
- **自定义classifiaction_trainer.py**： 原本是想使用transformer自带的trainer函数的，但是很多trick不支持，比如对抗训练，以及根据样本不均衡比例scale loss等，所以自己实现了一个。 
- **run\_\*.sh**: 在`./runs`下新建输出路径，路径名由NOWTIME和TASKNAME生成；同时指定配置文件的路径；根据自己的需要可以指定do_train do_eval等来选择训练、测试或者预测。

### 一般步骤
- 在data下新建一个存放该数据集的文件，比如这里是cnews，然后将数据放在该文件夹下。在文件夹里新建一个process.py文件，主要是获取标签并存储到labels.txt中。
  - 在`./dataset/dataset.py`文件中自定义一个类，可参考CnewsDataset，需要注意这里是将所有数据都先读到内存里进行处理的，如果数据集较大超过内存大小则修改。
  - 在`main.py` 中可以定义自己的lr_scheduler, optimizer等
  - 接着在`./conf`中新建一个yml文件，根据自己任务调整相应的参数配置
  - 最后运行对应的`run_*.sh`即可。


### 示例
#### 单标签分类 cnews数据集
数据下载地址：链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud  
git clone后执行命令：
```
cd ./text_classification
bash run_cnews.sh
```

运行了100步之后的结果：
【dev】 loss：323.154190  accuracy:0.860200  micro_f1:0.860200  macro_f1:0.856202
```
precision    recall  f1-score   support

          时政       0.93      0.95      0.94       500
          娱乐       0.98      0.78      0.87       500
          教育       0.89      0.62      0.73       500
          游戏       0.91      0.98      0.94       500
          时尚       0.70      0.99      0.82       500
          房产       0.77      0.81      0.79       500
          体育       0.99      1.00      1.00       500
          科技       0.92      0.97      0.95       500
          财经       0.96      0.99      0.97       500
          家居       0.59      0.51      0.55       500

    accuracy                           0.86      5000
   macro avg       0.87      0.86      0.86      5000
weighted avg       0.87      0.86      0.86      5000
```

#### 多标签分类 baidu2020数据集
[数据来源](https://github.com/taishan1994/pytorch_bert_multi_classification)
```
# 需要在main.py中修改下数据集类名，将cnewsDataset换为Baidu2020Dataset
# 因为这份数据集中没有单独的测试集，所以trainer中的dev_loader和test_loader都传dev_loader即可，或者test_loader传None也可。
# 修改完毕后，执行如下命令

cd ./text_classification
bash run_baidu2020.sh
```
运行了3000步之后的结果：
【dev】 loss：0.672807  accuracy:0.873832  micro_f1:0.927427  macro_f1:0.853027

```
              precision    recall  f1-score   support

 财经/交易-出售/收购       1.00      0.96      0.98        24
    财经/交易-跌停       1.00      0.93      0.96        14
    财经/交易-加息       0.00      0.00      0.00         3
    财经/交易-降价       0.86      0.67      0.75         9
    财经/交易-降息       0.00      0.00      0.00         4
    财经/交易-融资       1.00      1.00      1.00        14
    财经/交易-上市       1.00      0.43      0.60         7
    财经/交易-涨价       1.00      0.80      0.89         5
    财经/交易-涨停       1.00      1.00      1.00        27
     产品行为-发布       0.95      0.97      0.96       150
     产品行为-获奖       1.00      0.81      0.90        16
     产品行为-上映       1.00      0.86      0.92        35
     产品行为-下架       1.00      0.96      0.98        24
     产品行为-召回       1.00      1.00      1.00        36
       交往-道歉       0.94      0.79      0.86        19
       交往-点赞       1.00      0.91      0.95        11
       交往-感谢       1.00      1.00      1.00         8
       交往-会见       1.00      1.00      1.00        12
       交往-探班       1.00      0.80      0.89        10
     竞赛行为-夺冠       0.89      0.91      0.90        56
     竞赛行为-晋级       0.83      0.91      0.87        33
     竞赛行为-禁赛       1.00      0.88      0.93        16
     竞赛行为-胜负       0.97      1.00      0.98       213
     竞赛行为-退赛       0.94      0.83      0.88        18
     竞赛行为-退役       1.00      0.91      0.95        11
     人生-产子/女       1.00      0.80      0.89        15
       人生-出轨       1.00      0.25      0.40         4
       人生-订婚       1.00      0.89      0.94         9
       人生-分手       1.00      0.87      0.93        15
       人生-怀孕       1.00      0.62      0.77         8
       人生-婚礼       1.00      0.50      0.67         6
       人生-结婚       0.97      0.79      0.87        43
       人生-离婚       1.00      0.94      0.97        33
       人生-庆生       1.00      0.88      0.93        16
       人生-求婚       1.00      1.00      1.00         9
       人生-失联       1.00      0.64      0.78        14
       人生-死亡       0.93      0.83      0.88       106
     司法行为-罚款       1.00      0.90      0.95        29
     司法行为-拘捕       0.97      0.97      0.97        88
     司法行为-举报       1.00      1.00      1.00        12
     司法行为-开庭       1.00      0.93      0.96        14
     司法行为-立案       1.00      1.00      1.00         9
     司法行为-起诉       1.00      0.67      0.80        21
     司法行为-入狱       0.89      0.89      0.89        18
     司法行为-约谈       0.97      0.97      0.97        32
    灾害/意外-爆炸       1.00      0.67      0.80         9
    灾害/意外-车祸       0.91      0.89      0.90        35
    灾害/意外-地震       1.00      1.00      1.00        14
    灾害/意外-洪灾       1.00      0.43      0.60         7
    灾害/意外-起火       1.00      0.81      0.90        27
  灾害/意外-坍/垮塌       1.00      0.80      0.89        10
    灾害/意外-袭击       0.86      0.75      0.80        16
    灾害/意外-坠机       1.00      0.92      0.96        13
     组织关系-裁员       1.00      0.79      0.88        19
   组织关系-辞/离职       1.00      0.99      0.99        71
     组织关系-加盟       0.97      0.80      0.88        41
     组织关系-解雇       1.00      0.54      0.70        13
     组织关系-解散       1.00      1.00      1.00        10
     组织关系-解约       0.00      0.00      0.00         5
     组织关系-停职       1.00      0.82      0.90        11
     组织关系-退出       0.94      0.73      0.82        22
     组织行为-罢工       1.00      0.88      0.93         8
     组织行为-闭幕       1.00      0.78      0.88         9
     组织行为-开幕       0.97      0.94      0.95        32
     组织行为-游行       1.00      0.89      0.94         9

   micro avg       0.97      0.89      0.93      1657
   macro avg       0.93      0.80      0.85      1657
weighted avg       0.96      0.89      0.92      1657
 samples avg       0.93      0.91      0.92      1657

```
# 目前支持的功能：
- 对抗训练：仅支持FGM
- 混合精度训练
- 梯度累积
- 单标签分类、多标签分类
- ckpt 继续训练
- 单卡训练

# TODO:
- 支持更多样的自定义的模型结构：
  - [x] transformer类模型： pretrain + MLP
  - [ ] 传统机器学习浅层模型 (doing)
  - [ ] 深度学习模型
  - [ ] prompt learning 文本分类 (doing)
- 支持更多的功能：
  - [ ] 模型蒸馏
  - [ ] 转onnx (doing)
  - [ ] ckpt 继续训练 功能的完善
  - [ ] 更多文本分类trick支持：如focal loss
  - [ ] 分布式训练 (doing)
  - [ ] 读取数据集方式修改（目前是一次性读到内存中hh） (doing)


# 后续优化方向：
## PLM类
- 多尝试不同的预训练模型： RoBERT, WWM, ALBERT
- 除了用cls外还可以用avg, max池化做句向量的表示，甚至可以把不同层组合起来
- 在领域数据上做增量预训练
- 集成蒸馏， 训多个大模型集成起来然后蒸馏到一个上
- 先用多任务训，再迁移到自己的任务上

# 关于实际工作中的经验（待完善）
懂得同学应该能看明白，这个项目比较简单，也比较入门。但在实际工作中，不是随便一个分类任务套一下这套代码就能work的，要针对具体情况具体分析。
以下是一些积累的经验：
- **数据集构建**：
  - **标签体系构建**：拿到任务自己先标100条左右，看看哪些是难以确定的，如果占比太多，那这个任务定义就有问题，或者标签体系不清晰，要及时反馈。
  - **训练评估集的构建**： 构建两个评估集：一个是贴合真实数据分布的线上评估集，反映线上效果。 另一个是用规则去重后均匀采样的随机评估集，反映模型的真实能力。训练集尽可能和评估集分布一致，有时会用相近领域拿线程的有标注训练数据，这时要注意调整分布，比如句子长度、标点、干净程度等，尽可能做到自己分不出这个句子是本任务的还是别人那里借来的。
  - **数据清洗**：
    - **去掉文本强pattern**:  比如新闻语料中，很多数据包含xx报道，xx编辑等高频字段就没有用，可以把很高频的无用元素去掉；还有比如在判断无意义闲聊分类时，一个句号就可能让样本由正转负，这是因为训练预料中闲聊很少带句号，去掉这个pattern就好了。
    - **纠正标注错误**： 具体做法是 把训练集和评估集拼起来，用该数据训练模型两三个epoch,再去预测这个数据集，把模型判错的拿出来按照label-prob排序（比如label是1，模型预测是0.4，diff是0.6，diff很大，就按照这个diff排序），少的话就自己看，多的话就反馈给标注人员。
    - **去重**： 针对训练集要进行去重，避免过拟合，有的样本比较难，可以多训，才有在loss上加权重的方式。
- **长文本**：
  - 简单任务，直接用fasttext就可以达到不错的效果。
  - bert的话，最简单的方式是粗暴截断，比如只取 句首+句尾、 句首+tfidf筛几个词出来；或者每句都预测，最后对结果综合。
  - 魔改模型：XLNet, Roformer, Longformer
  - 离线任务的话还是建议跑全部，相信模型的编码能力。
- **少样本**
  - bert之后，很少收到数据不均衡或者过少的困扰，先无脑训一版
  - 如果样本在几百条，可以先把分类问题转化为匹配问题，或者用这种思想再去标一些高置信度的数据，或者用自监督、半监督的方法。
- **鲁棒性**：
  - 实际中，很多badcase很尴尬，明明正常的对了，但加了一个字就错了。
  - 这里可以直接使用一些粗暴的数据增强，加停用词 加标点 删词 同义词替换等，如果效果下降就把增强后的训练数据洗一下。
  - 或者采用对抗学习、对比学习之类的技巧提升，一般可以提升一个点。


来源：
- [深度学习文本分类模型综述+代码+技巧](https://zhuanlan.zhihu.com/p/349086747)

# 最后
欢迎大家多多提issue，多多补充
  