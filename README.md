数据清洗流程：

llm.py
1. 大模型筛选与旅游事项相关的文本。
2. 构建正则库，
3. 大模型判断文本完整性，补全缺失的文本内容。

cluster.py
4. 用MiniCPM-embedding模型将文本向量化，并且进行聚类操作。
5. 对聚类模型的输出进行人工筛选，找出符合条件的类别，并且标记为正向或负向样本。

fine_tuning.sh
6. 将标记好的样本整理为embedding-finetune的训练数据集，微调MiniCPM-embedding。

classification.py
7. 用6.的微调模型直接对3.的数据进行二分类任务，直接筛选出符合要求的数据。


10. 对7.输出的数据进行人工抽检（1%配置），如果准确率达90%及以上，则不再修改MiniCPM-embedding模型。
9. 如果准确率不合格，则对抽检数据进行人工标注，将标注数据添加到6.的训练数据中，对MiniCPM-embedding进行二次微调。