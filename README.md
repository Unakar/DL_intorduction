# DL_intorduction

## 第一讲
不讲(默认掌握)：loss, SGD, optimizer, lr, MLP, softmax...
今日内容概要：
1. 基于pytorch实现一个cifar10分类任务的baseline model
  - 特征工程初探(基于transforms的standarlize，Augmentation)，图像变换
  - 模型搭建(前向反向传播，Trainer Demo，卷积核，池化，norm)
  - Optimizer(SGD,Adam,Weight decay,余弦退火原理)
2. 对Bottleneck优化与进一步探究，加深理解
  - 随着网络层数加深，模型效果是否一定变好？(Resnet，跳跃连接)
  - 玄学调参设置：不同任务lr该是多少？batch选择对模型准确率影响？(我的经验)
  - 如何对数据做增强，使得小数据集上一样获得较高的模型性能？(甚至Datafree)
  - 如何让模型快速收敛？模型前后层如何保持学到的特征不遗忘？(Norm)
  - 玄学调参2：模型参数初始化设置？(Kaiming initialize & Xavier)
  - 同等网络层数下如何提取更多特征？(通道数翻倍？多头注意力？)
  - 玄学解释3：为什么神经网络可以work?(feature map的可解释性)
3. 有空再讲：
  - 如何在该数据集上达到最高准确率？尝试逼近SOTA
  - 有无办法，给定数据集后不写任何神经网络代码，自动完成该问题下模型架构搭建与超参数设置(太伟大了NAS)
  - 模型太笨重，推理速度慢？(太伟大了模型压缩，剪枝量化，甚至可以部署在单片机上)
  - 如何让另一个模型学到一个已经训练好的模型的知识？（无需重新训练，太伟大了模型蒸馏）
  - 更多更多的trick分享.....以及一点深度学习的哲学思考