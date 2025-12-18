# SimpleOneMoE
用于个人学习，初步理解MoE的流程（代码参考：https://zhuanlan.zhihu.com/p/676980004）
每部分的专家网络以及后面的门控网络都是分开训练的，各自在自己的epoch循环里训练。

最终运行结果：

    # Expert 1 Accuracy: 0.517
    # Expert 2 Accuracy: 0.497
    # Expert 3 Accuracy: 0.450
    # Mixture of Experts Accuracy: 0.640

一些问题：

1.关于MoE能否节省计算资源：

参考Switch Transformer有关于节省计算资源内容（为什么可以用更少的资源实现更大参数量的模型）。前向传播只需要计算TopK专家的输出，而不是所有专家的输出（例如 计算机视觉相关论文：https://dl.acm.org/doi/10.1145/3746027.3755026）。
也有计算全部专家网络，随后进行加权求和的（例如 计算机视觉相关论文：https://openaccess.thecvf.com/content/ICCV2025/html/Liao_GM-MoE_Low-Light_Enhancement_with_Gated-Mechanism_Mixture-of-Experts_ICCV_2025_paper.html）。
    
上面两篇论文，一种专家网络相同，训练权重不同；一种专家网络设计不同。

2.复杂的模型框架中MoE如何训练？损失函数如何确定？
    （还在学习调研...）
