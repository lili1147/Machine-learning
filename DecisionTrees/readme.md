DecisionTrees  决策树

1.scikit-learn中调用






2.  FindBestEntropy.py
  模拟决策树寻找最低熵，即分类后使分类更确定  
  寻找最佳的信息熵
  信息熵越低，系统越稳定，即分类结果越好，信息熵越高，系统越不稳定
  信息熵的计算公式   H = -p1 * log(p1) + (-p2 * log(p2)) + (p3* log(p3)) + ...    其中p1,p2,p3 分别是每一类所占的比例


  决策树中一个参数为 entropy  =====》 模拟在哪个维度，哪个值做为决策树的一个节点
