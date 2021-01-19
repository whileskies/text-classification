# text-classification
machine learning course project

使用朴素贝叶斯及softmax回归方法进行文本分类

相关文件：
1. text_preprocessing.py 文本预处理，将训练集与测试集转换为SOW与BOW模型等格式存储，保存在preprocessing_data目录
2. naive_bayes_bernoulli.py 基于Multi-variate Bernoulli model模型的朴素贝叶斯文本分类预测
3. naive_bayes.py 基于Multinomial event model模型的朴素贝叶斯文本分类预测
4. softmax_classification.py softmax分类器

准确率：

| 模型名称                                 | 准确率 |
| ---------------------------------------- | ------ |
| Multi-variate Bernoulli model 朴素贝叶斯 | 67.64% |
| Multinomial event 朴素贝叶斯             | 93.90% |
| Softmax based presence                   | 89.92% |
| Softmax based frequency                  | 93.90% |

