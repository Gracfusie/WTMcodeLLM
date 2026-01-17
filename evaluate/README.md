Detector 部分会检验水印方法对于模型的鲁棒性。
对于带水印模型生成的样本，Detector 会得到假设检验的 z-score，并据此判断模型是否带水印。

Detector 的运行框架是：输入样本和样本对应的水印参数，得到对应的 z-score。样本可能被攻击（例如：截断、重构、标准化）。

- prompt库：放在input文件夹里，用于generate。
- generate sample数据：放在output文件夹里，是detect完成的结果。

！注意，现在的flow，想要获得z score，必须重复地跑进行一次推理和水印生成，这可能有点浪费，因为跑LiveCodeBench进行推理本身也能生成一些结果。

analyze_outputs.py：从evaluate/output收集信息并统计/画图的框架。
output里是文件名格式化的json文件，文件名里的信息json里应该都有。
把output文件夹里全部的信息统计完之后，再进行画图操作：

1. json里面有z-score信息，对于每一个参数（算法+模型参数）都输出z-score的分布情况
2. 分析每一个参数在不同z-threshold下的TPR-FPR等信息。这里假设具有none-watermark的信息，即alg字段为none的json，作为无水印样本和有水印样本混合。

还未实现：

3. 分析不同参数的表现。
4. 分析攻击下的表现。