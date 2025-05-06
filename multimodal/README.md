# RPPA和HiSeq多模态整合方案

## 1. 代码整合方案

### 1.1 目录结构
```
multimodal/
├── data/
│   ├── raw/
│   │   ├── common/
│   │   │   └── survival_three_groups.csv
│   │   ├── hiseq/
│   │   │   └── expression_data.csv
│   │   └── rppa/
│   │       └── rppa_data.csv
│   ├── processed/
│   └── __init__.py
│   ├── rppa_data.py     # RPPA数据处理
│   └── hiseq_data.py    # HiSeq数据处理
├── models/
│   ├── __init__.py
│   ├── base_model.py    # 基础模型类
│   ├── rppa_model.py    # RPPA模型
│   └── hiseq_model.py   # HiSeq模型
├── training/
│   ├── __init__.py
│   └── trainer.py       # 训练器
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py     # 评估器
├── utils/
│   ├── __init__.py
│   ├── config.py        # 配置文件
│   └── logger.py        # 日志工具
└── main.py              # 主程序入口
```

### 1.2 数据模块整合
```python
# data/rppa_data.py
class RPPADataProcessor:
    def __init__(self, config):
        self.config = config
        
    def load_data(self):
        """加载RPPA数据"""
        df = pd.read_csv(self.config['rppa_with_survival'])
        df = df.dropna(subset=['survival_group_code'])
        return df
        
    def process_data(self, df):
        """处理RPPA数据"""
        # 从原有classify.py中提取的数据处理逻辑
        df_features = df.drop(columns=['survival_group_code', 'sample'])
        df_features = df_features.fillna(df_features.median())
        return df_features

# data/hiseq_data.py
class HiSeqDataProcessor:
    def __init__(self, config):
        self.config = config
        
    def load_data(self):
        """加载HiSeq数据"""
        df = pd.read_csv(self.config['default_with_survival'])
        df = df.dropna(subset=['survival_group_code'])
        return df
        
    def process_data(self, df):
        """处理HiSeq数据"""
        # 从原有classify.py中提取的数据处理逻辑
        df_features = df.drop(columns=['survival_group_code', 'sample'])
        df_features = df_features.fillna(df_features.median())
        return df_features
```

### 1.3 模型模块整合
```python
# models/base_model.py
class BaseModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
    def forward(self, x):
        raise NotImplementedError

# models/rppa_model.py
class RPPAModel(BaseModel):
    def __init__(self, input_size):
        super().__init__(input_size)
        # 从原有classify.py中提取的RPPA模型结构
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
        
    def forward(self, x):
        return self.layers(x)

# models/hiseq_model.py
class HiSeqModel(BaseModel):
    def __init__(self, input_size):
        super().__init__(input_size)
        # 从原有classify.py中提取的HiSeq模型结构
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
        
    def forward(self, x):
        return self.layers(x)
```

### 1.4 训练模块整合
```python
# training/trainer.py
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = WeightedCrossEntropyLoss(weight=config['class_weights'])
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config['initial_lr'],
            momentum=0.9,
            weight_decay=0.001
        )
        
    def train(self, train_loader, val_loader):
        """从原有classify.py中提取的训练逻辑"""
        for epoch in range(self.config['epochs']):
            self._train_epoch(train_loader)
            self._validate_epoch(val_loader)
            
    def _train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        for inputs, labels in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
    def _validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
```

### 1.5 评估模块整合
```python
# evaluation/evaluator.py
class Evaluator:
    def __init__(self, config):
        self.config = config
        
    def evaluate(self, model, test_loader):
        """从原有classify.py中提取的评估逻辑"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())
                
        return self._calculate_metrics(all_preds, all_labels)
        
    def _calculate_metrics(self, preds, labels):
        """计算评估指标"""
        return {
            'accuracy': accuracy_score(labels, preds),
            'classification_report': classification_report(
                labels, preds,
                target_names=['Group 1', 'Group 2', 'Group 3']
            ),
            'confusion_matrix': confusion_matrix(labels, preds)
        }
```

### 1.6 主程序整合
```python
# main.py
def main():
    # 加载配置
    config = load_config()
    
    # 初始化数据处理器
    rppa_processor = RPPADataProcessor(config)
    hiseq_processor = HiSeqDataProcessor(config)
    
    # 加载和处理数据
    rppa_data = rppa_processor.load_data()
    hiseq_data = hiseq_processor.load_data()
    
    rppa_features = rppa_processor.process_data(rppa_data)
    hiseq_features = hiseq_processor.process_data(hiseq_data)
    
    # 初始化模型
    rppa_model = RPPAModel(rppa_features.shape[1])
    hiseq_model = HiSeqModel(hiseq_features.shape[1])
    
    # 训练模型
    rppa_trainer = Trainer(rppa_model, config)
    hiseq_trainer = Trainer(hiseq_model, config)
    
    # 评估模型
    evaluator = Evaluator(config)
    
    # 保存结果
    save_results(rppa_model, hiseq_model, evaluator)

if __name__ == "__main__":
    main()
```

## 2. 整合步骤

1. 创建新的项目结构
2. 从原有代码中提取核心功能
3. 重构代码以适应新的结构
4. 添加必要的配置和工具类
5. 实现主程序逻辑

## 3. 后续优化

1. 添加多模态融合功能
2. 实现动态模态注册
3. 优化训练流程
4. 增强评估功能
5. 添加可视化工具

## 数据说明

### 公共数据
- 生存数据文件 `survival_three_groups.csv` 应放置在 `data/raw/common/` 目录下
- 该文件包含样本ID和对应的生存组信息
- 列名：sampleID, _PATIENT, OS, OS.time, DSS, DSS.time, DFI, DFI.time, PFI, PFI.time, Redaction, survival_group_code, survival_group_label, survival_group, survival_days

### HiSeq数据
- 原始基因表达数据应放置在 `data/raw/hiseq/expression_data.csv`
- 数据格式：CSV文件，第一行为基因名称，第一列为样本ID
- 数据维度：约400+样本 x 20000+基因

### RPPA数据
- 原始蛋白表达数据应放置在 `data/raw/rppa/rppa_data.csv`

### 数据预处理
使用 `data/hiseq_data.py` 和 `data/rppa_data.py` 中的处理器类进行数据处理：
1. 提取与生存数据匹配的样本
2. 添加生存组信息
3. 保存处理后的数据到 `data/processed/` 目录 

## 系统配置

系统使用统一的配置管理，所有配置项都定义在`utils/config.py`中。主要配置包括：

### 数据路径配置
所有数据文件路径都配置在`DATA_PATHS`字典中，便于统一管理：

```python
DATA_PATHS = {
    # 原始数据路径
    'raw_data_dir': ROOT_DIR / 'data' / 'raw',
    'processed_data_dir': ROOT_DIR / 'data' / 'processed',
    
    # 公共数据
    'survival_data': ROOT_DIR / 'data' / 'raw' / 'common' / 'survival_three_groups.csv',
    
    # HiSeq数据
    'hiseq_expression_data': ROOT_DIR / 'data' / 'raw' / 'hiseq' / 'expression_data.csv',
    'hiseq_processed_data': ROOT_DIR / 'data' / 'processed' / 'hiseq_processed.csv',
    
    # RPPA数据
    'rppa_data': ROOT_DIR / 'data' / 'raw' / 'rppa' / 'rppa_data.csv',
    'rppa_processed_data': ROOT_DIR / 'data' / 'processed' / 'rppa_processed.csv',
}
```

### 数据处理配置
数据处理的相关参数配置在`DATA_PROCESSING`字典中：

```python
DATA_PROCESSING = {
    # 是否应用数据平衡
    'apply_smote': True,
    
    # 特征选择配置
    'feature_selection': {
        'enabled': True,
        'max_features': 1000,  # 最大特征数
        'method': 'variance',  # 特征选择方法
    },
    
    # 数据转换配置
    'transformation': {
        'hiseq': {
            'scaler': 'robust',  # 缩放方法
            'fill_na': 'median', # 缺失值填充方法
        },
        'rppa': {
            'scaler': 'robust',
            'fill_na': 'median',
        }
    },
}
```

### 日志系统
系统提供了统一的日志记录功能，定义在`utils/logger.py`中：

```python
# 获取默认日志记录器
from multimodal.utils.logger import LOGGER

# 记录日志
LOGGER.info("这是一条信息")
LOGGER.warning("这是一条警告")
LOGGER.error("这是一条错误")

# 获取特定模块的日志记录器
from multimodal.utils.logger import get_logger
module_logger = get_logger("module_name")
```

### 处理后数据存放位置
- HiSeq数据处理后的文件保存在：`data/processed/hiseq_processed.csv`
- RPPA数据处理后的文件保存在：`data/processed/rppa_processed.csv`

## 使用方法

### 数据预处理

```python
from multimodal.data.hiseq_data import HiSeqDataProcessor
from multimodal.data.rppa_data import RPPADataProcessor

# 处理HiSeq数据
hiseq_processor = HiSeqDataProcessor()
hiseq_data = hiseq_processor.preprocess()

# 处理RPPA数据
rppa_processor = RPPADataProcessor()
rppa_data = rppa_processor.preprocess()
``` 