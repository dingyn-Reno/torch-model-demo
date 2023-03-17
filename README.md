# torch-model-demo

一个基于深度学习的pytorch训练框架，方便kaggle竞赛，临时项目以及科研等工作。

代码现已为pytorch2.0进行优化。

## 使用说明

1 将模型代码放入model文件夹
2 将数据加载代码放入feeder文件夹
3 配置优化器
4 设置损失函数
5 设置评分函数
6 设置数据保存
7 配置文件放入config文件夹

```bash
pip install -e torchlight
```

## 运行项目

本项目提供了一个使用LeNet对FashionMnist进行分类的示例，运行代码

```bash
python main.py --config config/config.yaml --device 0
```

## 启动tensorboard

```bash
tensorboard --logdir=work_dir/mnist/runs --bind_all
```

## 训练和测试

test:

```bash
python main.py --config config/config.yaml --phase test --save-score True --device 0 --weights ?
```

train:

```bash
python main.py --config ? --device 
```

## version 2.0

### compile

新版本为pytorch 2.0 的compile功能进行了优化。我们增加了compile参数，以选择是否开启compile，如果开启，则开启编译功能,默认为false。

torch.compile具有mode参数，参数拥有以下三种，默认为False：

- default：默认优化
- reduce-overhead: 减少模型开销并提升内存，可以帮助小模型提速
- max-autotune: 最大化提升模型速度

### AMP

我们为自动混合精度提供了支持，设置参数AMP为True可以开启自动混合精度，默认为False。

同时我们配置了参数Scaler，设置为True可以放大损失值来防止梯度的下溢，默认为False
