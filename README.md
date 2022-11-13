# torch-model-demo
A sample of the PyTorch deep learning script.
## 使用说明
1 将模型代码放入model文件夹
2 将数据加载代码放入feeder文件夹
3 配置优化器
4 设置损失函数
5 设置评分函数
6 设置数据保存
7 配置文件放入config文件夹
## 运行项目
本项目提供了一个使用LeNet对FashionMnist进行分类的示例，运行代码
```bash
python main.py --config config/config.yaml --device 0
```
## 启动tensorboard
```bash
tensorboard --logdir=work_dir/mnist/runs --bind_all
```
