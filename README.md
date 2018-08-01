# Deap-learning-simple-NN_4
神经网络优化-----正则化损失函数
在神经网络优化中，通过对损失函数进行正则化来缓解过拟合

方法：通过在损失函数中引入模型复杂度指标，利用给W加权值，弱化了训练数据的噪声

公式为：

loss = loss(y与y_) + regularizer * loss(w)
其中 loss(y与y_)为一般损失函数，可以为交叉熵损失函数，或者均方误差损失函数

regularizer为正则化权重

regularizer * loss(w) 在python中用一下代码实现：

tf.contrib.layers.l2_regularizer(regularizer)(w)

我的csdn链接，里面有详细讲解及运行效果截图：https://blog.csdn.net/congcong7267/article/details/81324467

