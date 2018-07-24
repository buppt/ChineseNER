# ChineseNER

这是最简单的一个命名实体识别BiLSTM+CRF模型。
直接用的<a href="https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html">pytorch tutorial</a>里的Bilstm+crf模型.

数据用的是玻森数据提供的中文命名实体识别数据，https://bosonnlp.com 这是官网，在数据下载里面有一个中文命名实体识别数据集。

先运行data_util.py处理数据，供模型使用。

然后使用train.py训练即可。由于使用的是cpu，而且也没有使用batch，所以训练速度比较慢。
