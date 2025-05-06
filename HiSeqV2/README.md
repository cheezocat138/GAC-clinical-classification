# RPPA 生存分析项目

## 项目概览

本项目是一个基于基因表达数据的生存分析系统，利用机器学习技术预测癌症患者的生存组别。项目主要功能包括：

- 从基因表达数据中提取关键特征
- 训练生存预测模型
- 对新样本进行生存组别预测
- 进行生存分析并生成Kaplan-Meier生存曲线

项目采用混合架构的深度学习模型，结合残差网络和注意力机制，能够处理高维基因表达数据并提供准确的生存预测结果。

## 数据访问

### 数据格式

项目使用的主要数据文件包括：

- **基因表达数据**：包含样本ID和基因表达值的CSV文件
  - 默认路径：`py/data_prossessing/HiSeqV2_with_survival.csv`
  - 转置格式：`py/data_prossessing/HiSeqV2_transposed.csv`
- **临床数据**：包含患者临床信息的CSV文件
  - 默认路径：`results/0/extracted_clinical_features.csv`

### 数据预处理

数据预处理流程包括：

1. 缺失值处理：使用中位数填充
2. 特征选择：结合互信息、ANOVA和随机森林方法
3. 特征增强：添加基于聚类的辅助特征
4. 类别不平衡处理：使用SMOTE算法平衡各生存组别的样本数量

## 模型开发

### 算法选择

项目采用深度学习与传统机器学习相结合的混合方法：

1. **主要预测模型**：基于PyTorch实现的`HybridSurvivalPredictor`
   - 结合残差网络结构和Squeeze-and-Excitation注意力机制
   - 多层架构设计，专为处理大规模基因表达数据优化

2. **特征选择**：
   - 互信息法(Mutual Information)
   - ANOVA F值
   - 随机森林特征重要性

3. **集成学习**：
   - K折交叉验证模型集成
   - 软投票策略整合多个模型的预测结果

### 关键超参数

模型训练的关键超参数包括：

```python
TRAINING_PARAMS = {
    'random_seed': 42,
    'n_splits': 5,          # 交叉验证折数
    'batch_size': 32,       # 批量大小
    'max_epochs': 150,      # 最大训练周期
    'initial_lr': 0.01,     # 初始学习率
    'min_lr': 0.0001,       # 最小学习率
    'max_lr': 0.01,         # 最大学习率
    'patience': 30,         # 早停耐心值
    'class_weights': [1.0, 1.0, 1.0],  # 类别权重
}

FEATURE_SELECTION_PARAMS = {
    'mi_k': 500,        # 互信息选择的特征数量
    'f_k': 500,         # ANOVA选择的特征数量
    'rfe_k': 300,       # 递归特征消除的特征数量
    'min_features': 300, # 最小特征数量
}

CLUSTER_PARAMS = {
    'n_clusters': 3,     # 聚类数量
    'random_state': 42,  # 随机种子
}
```

## 代码结构

项目主要Python文件及其功能：

- **classify.py**: 主要执行文件，包含训练和生存分析功能
- **config.py**: 配置文件，存储所有文件路径和参数设置
- **models.py**: 模型定义文件，包含神经网络结构和损失函数
- **utils.py**: 工具函数库，包含数据处理和可视化函数
- **feature_processor.py**: 特征处理类，确保训练和预测时使用一致的特征处理流程
- **predict.py**: 模型预测模块，用于加载模型并进行预测
- **survival_analysis.py**: 生存分析模块，用于生成生存曲线和统计分析
- **train_model.py**: 模型训练模块，包含完整训练流程
- **run_survival_analysis.py**: 命令行工具，用于执行训练、预测和分析任务

### 核心类与函数

- **HybridSurvivalPredictor**: 混合架构的生存预测模型
- **SEBlock**: Squeeze-and-Excitation注意力模块
- **ResidualBlock**: 残差网络模块
- **WeightedCrossEntropyLoss**: 加权交叉熵损失函数
- **FeatureProcessor**: 特征处理类
- **enhanced_feature_selection**: 增强型特征选择算法
- **add_cluster_features**: 聚类特征增强函数
- **train_survival_predictor**: 模型训练主函数
- **predict_survival_group**: 生存组别预测函数
- **run_survival_analysis**: 生存分析执行函数

## 部署说明

### 环境要求

项目依赖以下主要库：

- Python 3.6+
- PyTorch 1.7+
- pandas
- numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib
- seaborn
- lifelines (生存分析)

### 安装依赖

```bash
pip install torch pandas numpy scikit-learn imbalanced-learn matplotlib seaborn lifelines
```

### 使用方法

1. **训练模型**:

```bash
python py/run_survival_analysis.py train -d <数据文件路径> -o <输出目录>
```

2. **预测新样本**:

```bash
python py/run_survival_analysis.py predict -r <基因表达数据> -m <模型文件路径> -o <输出文件>
```

3. **执行生存分析**:

```bash
python py/run_survival_analysis.py analyze -r <基因表达数据> -c <临床数据> -m <模型文件路径> -o <输出目录>
```

4. **直接从classify.py运行**:

```bash
python py/classify.py
```

### 输出说明

训练后的模型和结果将保存在以下位置：

- 最终模型: `results/final/final_survival_predictor_model.pth`
- 集成模型: `results/final/ensemble_survival_predictor_model.pth`
- 混淆矩阵: `results/final/confusion_matrix.png`
- 交叉验证结果: `results/final/cv_training_results.png`
- 特征重要性: `results/final/feature_importance.png`

生存分析结果将保存在以下位置：

- 预测结果: `results/survival_analysis/predictions.csv`
- 生存曲线: `results/survival_analysis/kaplan_meier_curves.png`
- Log-rank检验结果: `results/survival_analysis/logrank_test_results.csv`

## 模型性能

模型通过5折交叉验证进行评估，评估指标包括：

- 准确率
- 分类报告(精确率、召回率、F1值)
- 混淆矩阵
- 生存曲线和Log-rank检验结果

## 维护与更新

- 定期更新特征选择方法以适应新数据
- 优化模型架构以提高预测性能
- 添加更多生存分析指标和可视化选项 