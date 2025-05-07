"""
多模态训练器模块

该模块提供用于训练和评估多模态融合模型的工具。
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from multimodal.utils.config import CONFIG
from multimodal.utils.logger import get_logger
from multimodal.data.multimodal_dataset import MultiModalDataProcessor
from multimodal.models.fusion_model import (
    EarlyFusionModel,
    LateFusionModel,
    AttentionFusionModel,
    CrossModalTransformerFusion,
    HybridMultiModalNet
)
from multimodal.models.enhanced_fusion_model import (
    ResidualFusionModel,
    AutoEncoderFusionModel,
    GatedFusionModel,
    SelfPacedFusionModel
)

# 获取日志记录器
logger = get_logger("MultiModalTrainer")

class MultiModalTrainer:
    """多模态模型训练器"""
    
    def __init__(self, model_type='attention', config=None):
        """
        初始化训练器
        
        参数:
            model_type (str): 融合模型类型
            config (dict): 配置字典
        """
        self.model_type = model_type
        self.config = config if config is not None else CONFIG
        
        # 设置随机种子
        torch.manual_seed(self.config['TRAINING_PARAMS']['random_seed'])
        np.random.seed(self.config['TRAINING_PARAMS']['random_seed'])
        
        # 初始化数据处理器
        self.data_processor = MultiModalDataProcessor(self.config)
        
        # 初始化设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = None
        self.hiseq_dim = None
        self.rppa_dim = None
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': [],
            'best_epoch': 0,
            'best_val_acc': 0,
        }
        
        # 创建输出目录
        os.makedirs(self.config['OUTPUT_DIRS']['model_dir'], exist_ok=True)
        os.makedirs(self.config['OUTPUT_DIRS']['results_dir'], exist_ok=True)
        os.makedirs(self.config['OUTPUT_DIRS']['logs_dir'], exist_ok=True)
    
    def _create_model(self):
        """
        创建融合模型
        
        返回:
            torch.nn.Module: 创建的模型
        """
        if self.hiseq_dim is None or self.rppa_dim is None:
            raise ValueError("特征维度未设置，请先调用prepare_data方法")
        
        hidden_size = 256
        dropout_rate = 0.3
        num_classes = 3
        
        logger.info(f"创建模型，输入维度: HiSeq={self.hiseq_dim}, RPPA={self.rppa_dim}, 类别数={num_classes}")
        
        # 根据模型类型创建相应模型
        if self.model_type == 'early':
            model = EarlyFusionModel(
                self.hiseq_dim, 
                self.rppa_dim, 
                hidden_size, 
                dropout_rate, 
                num_classes
            )
        elif self.model_type == 'late':
            model = LateFusionModel(
                self.hiseq_dim, 
                self.rppa_dim, 
                hidden_size, 
                dropout_rate, 
                num_classes
            )
        elif self.model_type == 'attention':
            model = AttentionFusionModel(
                self.hiseq_dim, 
                self.rppa_dim, 
                hidden_size, 
                dropout_rate, 
                num_classes
            )
        elif self.model_type == 'transformer':
            model = CrossModalTransformerFusion(
                self.hiseq_dim, 
                self.rppa_dim, 
                hidden_size, 
                num_heads=4, 
                num_layers=2, 
                dropout_rate=dropout_rate, 
                num_classes=num_classes
            )
        elif self.model_type == 'hybrid':
            model = HybridMultiModalNet(
                self.hiseq_dim, 
                self.rppa_dim, 
                hidden_size, 
                dropout_rate, 
                num_classes
            )
        elif self.model_type == 'residual':
            model = ResidualFusionModel(
                self.hiseq_dim, 
                self.rppa_dim, 
                hidden_size, 
                dropout_rate, 
                num_classes
            )
        elif self.model_type == 'autoencoder':
            model = AutoEncoderFusionModel(
                self.hiseq_dim, 
                self.rppa_dim, 
                hidden_size, 
                dropout_rate, 
                num_classes
            )
        elif self.model_type == 'gated':
            model = GatedFusionModel(
                self.hiseq_dim, 
                self.rppa_dim, 
                hidden_size, 
                dropout_rate, 
                num_classes
            )
        elif self.model_type == 'selfpaced':
            model = SelfPacedFusionModel(
                self.hiseq_dim, 
                self.rppa_dim, 
                hidden_size, 
                dropout_rate, 
                num_classes
            )
        else:
            raise ValueError(f"未知的模型类型: {self.model_type}")
        
        logger.info(f"创建{self.model_type}融合模型")
        return model.to(self.device)
    
    def prepare_data(self, test_size=0.2):
        """
        准备训练和测试数据
        
        参数:
            test_size (float): 测试集比例
            
        返回:
            tuple: (train_loader, test_loader)
        """
        logger.info("准备数据...")
        
        # 加载数据
        hiseq_features, rppa_features, labels = self.data_processor.load_data()
        
        if hiseq_features is None or rppa_features is None or labels is None:
            raise ValueError("数据加载失败")
        
        # 预处理数据
        hiseq_processed, rppa_processed, labels_processed = self.data_processor.preprocess_data(
            hiseq_features, rppa_features, labels
        )
        
        # 设置特征维度
        self.hiseq_dim = hiseq_processed.shape[1]
        self.rppa_dim = rppa_processed.shape[1]
        
        logger.info(f"特征维度: HiSeq={self.hiseq_dim}, RPPA={self.rppa_dim}")
        
        # 创建模型（如果尚未创建）
        if self.model is None:
            self.model = self._create_model()
        
        # 创建数据加载器
        train_loader, test_loader = self.data_processor.get_dataloaders(
            hiseq_processed, rppa_processed, labels_processed,
            test_size=test_size,
            batch_size=self.config['TRAINING_PARAMS']['batch_size']
        )
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        return train_loader, test_loader
    
    def train(self, train_loader=None, val_loader=None, num_epochs=None):
        """
        训练模型
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            num_epochs (int): 训练轮数
            
        返回:
            dict: 训练历史记录
        """
        if train_loader is None:
            train_loader = self.train_loader
        
        if val_loader is None:
            val_loader = self.test_loader
        
        if num_epochs is None:
            num_epochs = self.config['TRAINING_PARAMS']['max_epochs']
        
        # 确保模型已创建
        if self.model is None:
            raise ValueError("模型未创建，请先调用prepare_data方法")
        
        logger.info(f"开始训练{self.model_type}融合模型...")
        
        # 设置损失函数和优化器
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(self.config['TRAINING_PARAMS']['class_weights']).float().to(self.device)
        )
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['TRAINING_PARAMS']['initial_lr'],
            weight_decay=0.001
        )
        
        # 学习率调度器
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config['TRAINING_PARAMS']['max_lr'],
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        # 早停
        patience = self.config['TRAINING_PARAMS']['patience']
        best_val_acc = 0
        best_epoch = 0
        no_improve_epochs = 0
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (hiseq_data, rppa_data, targets) in enumerate(train_loader):
                # 将数据转移到设备
                hiseq_data = hiseq_data.to(self.device)
                rppa_data = rppa_data.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                
                if self.model_type == 'late':
                    outputs, _, _ = self.model(hiseq_data, rppa_data)
                else:
                    outputs = self.model(hiseq_data, rppa_data)
                
                loss = criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # 更新指标
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                # 打印进度
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    logger.info(
                        f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] " +
                        f"Loss: {train_loss/(batch_idx+1):.4f} " +
                        f"Acc: {100.*train_correct/train_total:.2f}% " +
                        f"LR: {scheduler.get_last_lr()[0]:.6f}"
                    )
            
            # 计算训练精度
            train_acc = train_correct / train_total
            train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_idx, (hiseq_data, rppa_data, targets) in enumerate(val_loader):
                    # 将数据转移到设备
                    hiseq_data = hiseq_data.to(self.device)
                    rppa_data = rppa_data.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 前向传播
                    if self.model_type == 'late':
                        outputs, _, _ = self.model(hiseq_data, rppa_data)
                    else:
                        outputs = self.model(hiseq_data, rppa_data)
                    
                    loss = criterion(outputs, targets)
                    
                    # 更新指标
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # 计算验证精度
            val_acc = val_correct / val_total
            val_loss = val_loss / len(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(scheduler.get_last_lr()[0])
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} " +
                f"Train Loss: {train_loss:.4f} " +
                f"Train Acc: {100.*train_acc:.2f}% " +
                f"Val Loss: {val_loss:.4f} " +
                f"Val Acc: {100.*val_acc:.2f}%"
            )
            
            # 检查是否为最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                no_improve_epochs = 0
                
                # 保存最佳模型
                self._save_model(f"{self.model_type}_best_model.pth")
                logger.info(f"保存最佳模型，验证精度: {100.*best_val_acc:.2f}%")
            else:
                no_improve_epochs += 1
            
            # 检查早停
            if no_improve_epochs >= patience:
                logger.info(f"{patience}轮未改善，早停")
                break
        
        # 保存训练历史
        self.history['best_epoch'] = best_epoch
        self.history['best_val_acc'] = best_val_acc
        
        # 绘制训练曲线
        self._plot_training_history()
        
        logger.info(f"训练完成，最佳验证精度: {100.*best_val_acc:.2f}% (轮次 {best_epoch+1})")
        
        return self.history
    
    def evaluate(self, test_loader=None):
        """
        评估模型性能
        
        参数:
            test_loader (DataLoader): 测试数据加载器
            
        返回:
            dict: 评估结果
        """
        if test_loader is None:
            test_loader = self.test_loader
        
        # 确保模型已创建
        if self.model is None:
            raise ValueError("模型未创建，请先调用prepare_data方法")
        
        logger.info(f"评估{self.model_type}融合模型...")
        
        # 加载最佳模型（如果存在）
        best_model_path = os.path.join(
            self.config['MODEL_PATHS']['model_dir'],
            f"{self.model_type}_best_model.pth"
        )
        
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            logger.info(f"加载最佳模型: {best_model_path}")
        
        # 切换到评估模式
        self.model.eval()
        
        # 收集预测结果
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (hiseq_data, rppa_data, targets) in enumerate(test_loader):
                # 将数据转移到设备
                hiseq_data = hiseq_data.to(self.device)
                rppa_data = rppa_data.to(self.device)
                
                # 前向传播
                if self.model_type == 'late':
                    outputs, _, _ = self.model(hiseq_data, rppa_data)
                else:
                    outputs = self.model(hiseq_data, rppa_data)
                
                # 获取预测结果
                _, predicted = outputs.max(1)
                
                # 收集结果
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        # 计算各种指标
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='macro'
        )
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        class_report = classification_report(
            all_targets, all_predictions, 
            target_names=['Group 1', 'Group 2', 'Group 3'],
            output_dict=True
        )
        
        # 记录结果
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        # 打印结果
        logger.info(f"测试精度: {100.*accuracy:.2f}%")
        logger.info(f"宏平均精确率: {precision:.4f}")
        logger.info(f"宏平均召回率: {recall:.4f}")
        logger.info(f"宏平均F1分数: {f1:.4f}")
        logger.info(f"混淆矩阵:\n{conf_matrix}")
        logger.info(f"分类报告:\n{classification_report(all_targets, all_predictions, target_names=['Group 1', 'Group 2', 'Group 3'])}")
        
        # 保存结果
        self._save_evaluation_results(results)
        
        return results
    
    def _save_model(self, filename):
        """
        保存模型
        
        参数:
            filename (str): 文件名
        """
        model_path = os.path.join(self.config['MODEL_PATHS']['model_dir'], filename)
        torch.save(self.model.state_dict(), model_path)
    
    def _save_evaluation_results(self, results):
        """
        保存评估结果
        
        参数:
            results (dict): 评估结果
        """
        results_path = os.path.join(
            self.config['OUTPUT_DIRS']['results_dir'],
            f"{self.model_type}_evaluation_results.txt"
        )
        
        with open(results_path, 'w') as f:
            f.write(f"模型: {self.model_type}\n")
            f.write(f"准确率: {100.*results['accuracy']:.2f}%\n")
            f.write(f"宏平均精确率: {results['precision']:.4f}\n")
            f.write(f"宏平均召回率: {results['recall']:.4f}\n")
            f.write(f"宏平均F1分数: {results['f1']:.4f}\n")
            f.write(f"混淆矩阵:\n{results['confusion_matrix']}\n")
            f.write(f"分类报告:\n")
            # 将分类报告字典转换为字符串格式
            report = results['classification_report']
            if isinstance(report, dict):
                # 格式化字典报告
                f.write("              precision    recall  f1-score   support\n\n")
                for label, metrics in report.items():
                    if label in ['accuracy', 'macro avg', 'weighted avg']:
                        continue
                    f.write(f"{label:>10}       {metrics['precision']:.2f}      {metrics['recall']:.2f}      {metrics['f1-score']:.2f}      {int(metrics['support'])}\n")
                
                f.write(f"\n    accuracy                           {report['accuracy']:.2f}        {int(report['macro avg']['support'])}\n")
                f.write(f"   macro avg       {report['macro avg']['precision']:.2f}      {report['macro avg']['recall']:.2f}      {report['macro avg']['f1-score']:.2f}      {int(report['macro avg']['support'])}\n")
                f.write(f"weighted avg       {report['weighted avg']['precision']:.2f}      {report['weighted avg']['recall']:.2f}      {report['weighted avg']['f1-score']:.2f}      {int(report['weighted avg']['support'])}\n")
            else:
                # 报告已经是字符串
                f.write(str(report))
    
    def _plot_training_history(self):
        """绘制训练历史曲线"""
        plt.figure(figsize=(12, 10))
        
        # 绘制损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.axvline(x=self.history['best_epoch'], color='r', linestyle='--')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.title(f"{self.model_type} 模型训练损失")
        
        # 绘制精度曲线
        plt.subplot(2, 1, 2)
        plt.plot(self.history['train_acc'], label='训练精度')
        plt.plot(self.history['val_acc'], label='验证精度')
        plt.axvline(x=self.history['best_epoch'], color='r', linestyle='--')
        plt.xlabel('轮次')
        plt.ylabel('精度')
        plt.legend()
        plt.title(f"{self.model_type} 模型训练精度")
        
        # 保存图形
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config['OUTPUT_DIRS']['plots_dir'], f"{self.model_type}_training_history.png"),
            dpi=self.config['PLOT_PARAMS']['dpi']
        )
        plt.close()
    
    def compare_models(self, model_types=None, num_epochs=5):
        """
        比较不同类型的融合模型性能
        
        参数:
            model_types (list): 要比较的模型类型列表
            num_epochs (int): 每个模型的训练轮数
            
        返回:
            pd.DataFrame: 比较结果
        """
        if model_types is None:
            model_types = ['early', 'late', 'attention', 'transformer', 'hybrid']
        
        logger.info(f"比较模型类型: {', '.join(model_types)}")
        
        # 准备数据（只需要做一次）
        train_loader, test_loader = self.prepare_data()
        
        results = []
        
        # 依次训练和评估每种模型
        for model_type in model_types:
            logger.info(f"\n{'='*20} 训练{model_type}模型 {'='*20}")
            
            # 保存当前模型类型
            current_model_type = self.model_type
            
            # 设置新的模型类型
            self.model_type = model_type
            self.model = self._create_model()
            
            # 训练模型
            start_time = time.time()
            self.train(train_loader, test_loader, num_epochs=num_epochs)
            train_time = time.time() - start_time
            
            # 评估模型
            eval_results = self.evaluate(test_loader)
            
            # 收集结果
            model_result = {
                'model_type': model_type,
                'accuracy': eval_results['accuracy'],
                'precision': eval_results['precision'],
                'recall': eval_results['recall'],
                'f1': eval_results['f1'],
                'training_time': train_time,
                'parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
            
            results.append(model_result)
            
            # 恢复原始模型类型
            self.model_type = current_model_type
        
        # 创建比较结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存比较结果
        results_path = os.path.join(
            self.config['OUTPUT_DIRS']['results_dir'],
            'model_comparison_results.csv'
        )
        results_df.to_csv(results_path, index=False)
        
        # 绘制比较图表
        self._plot_comparison_results(results_df)
        
        logger.info("\n比较结果:")
        logger.info(results_df)
        
        return results_df
    
    def _plot_comparison_results(self, results_df):
        """
        绘制模型比较结果
        
        参数:
            results_df (pd.DataFrame): 比较结果
        """
        import seaborn as sns
        
        # 设置图表风格
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 准确率比较
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='model_type', y='accuracy', data=results_df)
        
        # 在柱子上显示数值
        for i, p in enumerate(ax.patches):
            ax.text(p.get_x() + p.get_width()/2., p.get_height(), 
                    f'{results_df["accuracy"].iloc[i]:.4f}',
                    ha='center', va='bottom')
            
        plt.title('不同融合模型的准确率比较', fontsize=14)
        plt.xlabel('融合模型类型', fontsize=12)
        plt.ylabel('准确率', fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config['OUTPUT_DIRS']['plots_dir'], 'accuracy_comparison.png'),
            dpi=self.config['PLOT_PARAMS']['dpi']
        )
        
        # 训练时间比较
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='model_type', y='training_time', data=results_df)
        
        # 在柱子上显示数值
        for i, p in enumerate(ax.patches):
            ax.text(p.get_x() + p.get_width()/2., p.get_height(), 
                    f'{results_df["training_time"].iloc[i]:.1f}s',
                    ha='center', va='bottom')
            
        plt.title('不同融合模型的训练时间比较', fontsize=14)
        plt.xlabel('融合模型类型', fontsize=12)
        plt.ylabel('训练时间(秒)', fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config['OUTPUT_DIRS']['plots_dir'], 'training_time_comparison.png'),
            dpi=self.config['PLOT_PARAMS']['dpi']
        )
        
        # 参数量比较
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='model_type', y='parameters', data=results_df)
        
        # 在柱子上显示数值
        for i, p in enumerate(ax.patches):
            ax.text(p.get_x() + p.get_width()/2., p.get_height(), 
                    f'{results_df["parameters"].iloc[i]:,}',
                    ha='center', va='bottom')
            
        plt.title('不同融合模型的参数量比较', fontsize=14)
        plt.xlabel('融合模型类型', fontsize=12)
        plt.ylabel('参数量', fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config['OUTPUT_DIRS']['plots_dir'], 'parameters_comparison.png'),
            dpi=self.config['PLOT_PARAMS']['dpi']
        )
        
        # 多指标比较雷达图
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_types = results_df['model_type'].tolist()
        
        # 创建雷达图
        plt.figure(figsize=(10, 8))
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        ax = plt.subplot(111, polar=True)
        
        for i, model in enumerate(model_types):
            values = results_df.loc[i, metrics].tolist()
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        plt.xticks(angles[:-1], metrics)
        plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8'])
        plt.ylim(0.3, 0.8)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('不同融合模型在各指标上的表现', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config['OUTPUT_DIRS']['plots_dir'], 'metrics_radar_chart.png'),
            dpi=self.config['PLOT_PARAMS']['dpi']
        )
        
        plt.close('all') 