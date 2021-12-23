

# Graph Attention Network 

作者：petar velickovic，==Yoshua Bengio==

单位：剑桥大学，MILA

会议：ICLR’2018

## 卷积中的时域、空域

- spectral
- non-spectral

![preview](Graph Attention Network.assets/v2-340ea7d36319c985b7d7606b942ed4c9_r.jpg)

GAT：是基于空域的模型

<img src="Graph Attention Network.assets/image-20211223201819989.png" alt="image-20211223201819989" style="zoom:67%;" />

左边：空域  

右边：频域

## 论文结构

![image-20211223192313600](Graph Attention Network.assets/image-20211223192313600.png)

### 图像上的卷积操作

<img src="Graph Attention Network.assets/o_image-13-conv-cnn.gif" alt="o_image-13-conv-cnn" style="zoom:67%;" />

### 图上的卷积操作

<img src="Graph Attention Network.assets/image-20211223200441630.png" alt="image-20211223200441630" style="zoom:67%;" />

- GCN:（ha+hb+hc+hd）/   4
- GraphSage:   maxpooling(ha,hb,hc,hd)
- GAT:  计算两个点之间的相似度，如果两个点之间相似度高，赋予权值越大

### 计算图

<img src="Graph Attention Network.assets/image-20211223201102717.png" alt="image-20211223201102717" style="zoom:67%;" />

### 直推式(transductive)/归纳式(inductive learning)

不同于传统的图学习——都是直推式（相当于对训练集和测试集都进行训练）

GCN、GAT、GraphSage都可用于归纳式学习

### GAT中的multi-head

![image-20211223202155147](Graph Attention Network.assets/image-20211223202155147.png)

### 算法细节

$input: h={\vec{h_1},\vec{h_2},...,\vec{h_n}}$

$\vec{h_i} \in R^F$ : 即每个**hi**的维度是F

$output: h’={\vec{h'_1},\vec{h'_2},...,\vec{h'_n}}$

$\vec{h'_i} \in R^{F'}$ : 即每个**h’i**的维度是F’

