Detector 部分会检验水印方法对于模型的鲁棒性。
对于带水印模型生成的样本，Detector 会得到假设检验的 z-score，并据此判断模型是否带水印。

Detector 的运行框架是：输入样本和样本对应的水印参数，得到对应的 z-score。样本可能被攻击（例如：截断、重构、标准化）。

prompt库：放在input文件夹里，用于generate。
generate sample数据：放在output文件夹里，用于detect。