# LeNet

[原文地址](https://blog.csdn.net/qq_42570457/article/details/81460807)

输入的二维图像，先经过两次卷积层到池化层，再经过全连接层，最后使用softmax分类作为输出层。

是一种用于手写体字符识别的非常高效的卷积神经网络
![](../../../../MONTY_~1/AppData/Local/Temp/dl_3_1.png)

## INPUT层-输入层
输入图像的尺寸同一规划为32*32 ~~_(一般不将输入层视为网络层次结构之一)_~~
## C1层-卷积层
* 输入图片：32*32
* 卷积核大小：5*5
* 卷积核种类：6
* 输出 featuremap 大小：28*28（32-5+1）
* 神经元数量：28 * 28 * 6
* 可训练参数：（5*5+1）*6 （每个滤波器5*5=25个unit参数和一个bias参数，一共6个滤波器）
* 连接数：（5 * 5+1）* 6 * 28 * 28=122304

## S2层-池化层（下采样层）
## C3层-卷积层
## S4层-卷积层
## C5层-卷积层
## F6层-全连接层
## Output层-全连接层

# AlexNet
[原文地址](https://zhuanlan.zhihu.com/p/42914388)

AlexNet 共有8层结构，前5层为卷积层，后三层为全连接层。

![](../../../../MONTY_~1/AppData/Local/Temp/v2-3f5a7ab9bcb15004d5a08fdf71e6a775_720w.png)


# VGGNet


# InceptionNet
# ResNet