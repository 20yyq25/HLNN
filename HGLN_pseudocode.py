"""
HGLN (Hyperbolic Graph Legendre Network) 伪代码
===============================================
本文件展示了HGLN模型的整体架构和运行流程的伪代码
"""

# ============================================================================
# 1. 数据加载 (utils/data_utils.py)
# ============================================================================

def load_data(args, datapath):
    """
    加载图数据
    """
    if args.task == 'nc':  # 节点分类任务
        adj, features, labels, idx_train, idx_val, idx_test = load_data_nc(...)
    else:  # 链接预测任务
        adj, features = load_data_lp(...)
        # 划分训练/验证/测试边
    
    # 数据预处理
    adj_train_norm = normalize_adj(adj)  # 归一化邻接矩阵
    features = normalize_features(features)  # 归一化特征
    
    return {
        'adj_train_norm': adj_train_norm,
        'features': features,
        'labels': labels,  # 节点分类任务
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test
    }


# ============================================================================
# 2. 模型初始化 (models/base_models.py + models/encoders.py)
# ============================================================================

class HGLN_Model:
    """
    HGLN模型主类
    """
    def __init__(args):
        # 2.1 初始化流形 (manifolds/hyperboloid.py)
        self.manifold = HyperboloidManifold()
        self.c = args.c  # 双曲曲率参数
        
        # 2.2 初始化编码器 (models/encoders.py -> HGLN类)
        self.encoder = HGLN_Encoder(c, args)
        
        # 2.3 初始化解码器 (models/decoders.py -> LinearDecoder)
        self.decoder = LinearDecoder(c, args)
    
    def encode(x, adj):
        """
        编码：将节点特征编码为嵌入向量
        """
        # 如果是Hyperboloid流形，需要添加一个维度
        if manifold.name == 'Hyperboloid':
            x = add_zero_dimension(x)  # [N, d] -> [N, d+1]
        
        # 投影到切空间
        x_tan = manifold.proj_tan0(x, c=curvatures[0])
        
        # 指数映射到双曲空间
        x_hyp = manifold.expmap0(x_tan, c=curvatures[0])
        x_hyp = manifold.proj(x_hyp, c=curvatures[0])
        
        # 通过HGLN编码器层
        embeddings = encoder.forward(x_hyp, adj)
        
        return embeddings


# ============================================================================
# 3. HGLN编码器 (models/encoders.py -> HGLN类)
# ============================================================================

class HGLN_Encoder:
    """
    HGLN编码器：多层HyperbolicLegendreNet
    """
    def __init__(c, args):
        self.manifold = HyperboloidManifold()
        self.curvatures = [c1, c2, ..., c_final]  # 每层的曲率
        
        # 构建多层网络
        layers = []
        for i in range(num_layers - 1):
            layer = HyperbolicLegendreNet(
                manifold=manifold,
                in_features=dims[i],
                out_features=dims[i+1],
                c=c,
                c_in=curvatures[i],
                c_out=curvatures[i+1],
                k=args.k,  # 传播步数
                dropout=args.dropout,
                dprate=args.dprate,
                act=args.act
            )
            layers.append(layer)
        
        self.layers = Sequential(*layers)
    
    def forward(x, adj):
        """
        前向传播
        """
        for layer in self.layers:
            x, adj = layer.forward((x, adj))
        return x


# ============================================================================
# 4. HyperbolicLegendreNet层 (layers/hyp_layers.py)
# ============================================================================

class HyperbolicLegendreNet:
    """
    双曲勒让德网络层
    """
    def __init__(manifold, in_features, out_features, c, c_in, c_out, k, dropout, dprate, act, bias):
        # 4.1 双曲线性变换
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, bias)
        
        # 4.2 双曲激活
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        
        # 4.3 Bernoulli传播层 (核心组件)
        self.prop = Bern_prop(K=k, c=c, manifold=manifold, adaptive=True)
    
    def forward(input):
        """
        前向传播流程
        """
        x, adj = input
        
        # Step 1: Dropout
        h = dropout(x, p=dropout)
        
        # Step 2: 双曲线性变换
        h = self.linear.forward(h)  # 在双曲空间中的矩阵乘法
        
        # Step 3: 双曲激活
        h = self.hyp_act.forward(h)
        
        # Step 4: Bernoulli传播（使用勒让德多项式）
        edge_index = adj.coalesce().indices()  # 转换为边索引格式
        h = self.prop.forward(h, edge_index)  # 核心传播步骤
        
        if dprate > 0:
            h = dropout(h, p=dprate)
        
        return h, adj


# ============================================================================
# 5. Bern_prop传播层 (layers/Bernpro.py) - 核心组件
# ============================================================================

class Bern_prop:
    """
    Bernoulli传播层：使用勒让德多项式进行图传播
    """
    def __init__(K, c, manifold, adaptive=True):
        self.K = K  # 最大传播阶数
        self.c = c  # 双曲曲率
        self.manifold = manifold
        self.adaptive = adaptive
        
        if adaptive:
            # 自适应阶数参数
            self.alpha = Parameter(torch.Tensor([1.0]))
            self.beta = Parameter(torch.Tensor([0.5]))
        
        # 温度参数（用于勒让德多项式系数）
        self.temp = Parameter(torch.Tensor([1.0] * (K + 1)))
    
    def get_adaptive_K(self, x):
        """
        计算自适应传播阶数
        """
        if not self.adaptive:
            return self.K
        
        # 使用sigmoid确保K在合理范围内
        adaptive_K = sigmoid(alpha) * self.K + sigmoid(beta) * 2
        return int(adaptive_K.item())
    
    def forward(x, edge_index):
        """
        Bernoulli传播前向传播
        """
        # Step 1: 计算自适应阶数
        current_K = self.get_adaptive_K(x)  # 根据输入自适应调整K
        
        # Step 2: 构造拉普拉斯矩阵
        L, L_mod = construct_laplacian(edge_index, num_nodes)
        # L: 标准拉普拉斯矩阵
        # L_mod: 修改后的拉普拉斯矩阵（添加自环）
        
        # Step 3: 勒让德多项式展开
        tmp = [x]  # 存储每一阶的传播结果
        
        # 使用当前阶数进行传播
        for i in range(current_K):
            # 在双曲空间中进行矩阵向量乘法
            x = manifold.mobius_matvec(L_mod, x, c)
            tmp.append(x)
        
        # Step 4: 勒让德多项式求和
        # 计算勒让德多项式系数并加权求和
        out = (1/factorial(0)) * (1/(2^0)) * temp[0] * tmp[0]
        
        for i in range(current_K):
            x = tmp[i + 1]
            # 应用拉普拉斯算子
            x = manifold.mobius_matvec(L, x, c)
            for j in range(i):
                x = manifold.mobius_matvec(L, x, c)
            
            # 计算勒让德多项式系数
            coeff = (-1)^(i+1) * (1/factorial(i+1)) * comb(2*(i+1), i+1)
            out = out + coeff * temp[i+1] * x
        
        return out


# ============================================================================
# 6. 解码器 (models/decoders.py -> LinearDecoder)
# ============================================================================

class LinearDecoder:
    """
    线性解码器：用于节点分类
    """
    def decode(embeddings, adj):
        """
        解码：将嵌入向量映射到类别
        """
        # 从双曲空间投影回切空间
        h_tan = manifold.proj_tan0(manifold.logmap0(embeddings, c), c)
        
        # 线性分类
        logits = linear_layer(h_tan)
        
        return log_softmax(logits)


# ============================================================================
# 7. 训练流程 (train.py)
# ============================================================================

def train(args):
    """
    训练主函数
    """
    # 7.1 设置随机种子
    set_seed(args.seed)
    
    # 7.2 加载数据
    data = load_data(args, datapath)
    
    # 7.3 初始化模型
    if args.task == 'nc':
        model = NCModel(args)  # 节点分类模型
    else:
        model = LPModel(args)  # 链接预测模型
    
    # 7.4 初始化优化器
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_reduce_freq, gamma=args.gamma)
    
    # 7.5 训练循环
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        
        # 计算损失
        metrics = model.compute_metrics(embeddings, data, 'train')
        loss = metrics['loss']
        
        # 反向传播
        loss.backward()
        if args.grad_clip:
            clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        
        # 验证
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            
            # 早停检查
            if has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter >= args.patience:
                    break  # 早停
    
    return best_test_metrics


# ============================================================================
# 8. 关键数学操作（双曲空间）
# ============================================================================

"""
双曲空间中的关键操作（Hyperboloid流形）：

1. 投影到切空间：
   proj_tan0(x, c) -> 将点投影到原点的切空间

2. 指数映射：
   expmap0(v, c) -> 将切空间向量映射到双曲空间

3. 对数映射：
   logmap0(x, c) -> 将双曲空间点映射回切空间

4. Möbius矩阵向量乘法：
   mobius_matvec(W, x, c) -> 在双曲空间中执行 W @ x

5. Möbius加法：
   mobius_add(x, y, c) -> 在双曲空间中执行 x + y

6. 投影：
   proj(x, c) -> 确保点在双曲流形上
"""


# ============================================================================
# 9. 数据流示例
# ============================================================================

"""
完整数据流：

输入特征 [N, d]
    ↓
添加零维度（Hyperboloid）[N, d+1]
    ↓
投影到切空间 [N, d+1]
    ↓
指数映射到双曲空间 [N, d+1]
    ↓
┌─────────────────────────────────────┐
│  HGLN编码器（多层）                  │
│  ┌───────────────────────────────┐  │
│  │ HyperbolicLegendreNet Layer 1 │  │
│  │  - HypLinear                  │  │
│  │  - HypAct                     │  │
│  │  - Bern_prop (K阶传播)        │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ HyperbolicLegendreNet Layer 2 │  │
│  │  ...                          │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    ↓
嵌入向量 [N, dim]
    ↓
解码器（LinearDecoder）
    ↓
类别概率 [N, n_classes]
"""


# ============================================================================
# 10. 关键参数说明
# ============================================================================

"""
HGLN关键参数：

- K (k): 传播阶数，控制Bernoulli传播的深度
- c: 双曲曲率参数，控制双曲空间的弯曲程度
- adaptive: 是否使用自适应阶数（根据输入动态调整K）
- num_layers: 编码器层数
- dim: 嵌入维度
- dropout: Dropout概率
- dprate: 传播层的Dropout概率
- manifold: 流形类型（Hyperboloid/PoincareBall/Euclidean）

注意：已移除聚合（HypAgg）部分，图结构信息通过Bernoulli传播层处理
"""

