```mermaid
graph LR
  subgraph A
    B(数据基本处理)
    B --> C(特征工程)
    C --> D(机器学习)
    D --> E(模型评估)
    E --> B
  end
  
  F(用户数据) --> A
  
  A --> G(在线服务)
  
```
