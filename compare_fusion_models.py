"""
融合模型比较脚本

该脚本比较多种融合模型在相同数据集上的性能表现。
"""

import sys
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
sys.path.append('.')

from multimodal.training.multimodal_trainer import MultiModalTrainer
from multimodal.utils.config import CONFIG

# 设置随机种子确保可重复性
import torch
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def train_and_evaluate(model_type, num_epochs=5, verbose=True):
    """训练并评估指定类型的融合模型"""
    if verbose:
        print(f"\n\n{'='*20} 训练 {model_type} 模型 {'='*20}")
    
    # 初始化训练器
    trainer = MultiModalTrainer(model_type=model_type)
    
    # 准备数据
    train_loader, val_loader = trainer.prepare_data()
    
    # 训练模型
    start_time = time.time()
    history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    training_time = time.time() - start_time
    
    # 评估模型
    eval_results = trainer.evaluate(val_loader)
    
    # 计算模型参数量
    model_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    
    # 收集结果
    results = {
        'model_type': model_type,
        'accuracy': eval_results['accuracy'],
        'precision': eval_results.get('classification_report', {}).get('macro avg', {}).get('precision', 0),
        'recall': eval_results.get('classification_report', {}).get('macro avg', {}).get('recall', 0),
        'f1': eval_results.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0),
        'training_time': training_time,
        'parameters': model_params,
        'best_epoch': trainer.history['best_epoch'],
        'best_val_acc': trainer.history['best_val_acc']
    }
    
    if verbose:
        print(f"\n{'='*20} {model_type} 模型评估完成 {'='*20}")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"训练时间: {results['training_time']:.2f}秒")
        print(f"参数量: {results['parameters']:,}")
        print(f"最佳验证准确率: {results['best_val_acc']:.4f} (轮次 {results['best_epoch']})")
    
    return results

def compare_models(model_types=['early', 'late', 'attention', 'transformer', 'hybrid'], 
                  num_epochs=5, results_dir=None):
    """比较多种融合模型的性能"""
    if results_dir is None:
        results_dir = CONFIG['OUTPUT_DIRS']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    
    for model_type in model_types:
        model_results = train_and_evaluate(model_type, num_epochs)
        results.append(model_results)
    
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果
    results_file = os.path.join(results_dir, 'model_comparison_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n结果已保存到: {results_file}")
    
    # 可视化结果
    plot_comparison_results(results_df, results_dir)
    
    return results_df

def plot_comparison_results(results_df, results_dir):
    """绘制模型比较结果"""
    # 设置图表风格
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 准确率比较
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_type', y='accuracy', data=results_df)
    plt.title('不同融合模型的准确率比较', fontsize=14)
    plt.xlabel('融合模型类型', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'), dpi=300)
    
    # 训练时间比较
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_type', y='training_time', data=results_df)
    plt.title('不同融合模型的训练时间比较', fontsize=14)
    plt.xlabel('融合模型类型', fontsize=12)
    plt.ylabel('训练时间(秒)', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_time_comparison.png'), dpi=300)
    
    # 参数量比较
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_type', y='parameters', data=results_df)
    plt.title('不同融合模型的参数量比较', fontsize=14)
    plt.xlabel('融合模型类型', fontsize=12)
    plt.ylabel('参数量', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'parameters_comparison.png'), dpi=300)
    
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
    plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8], ['0.4', '0.5', '0.6', '0.7', '0.8'])
    plt.ylim(0.4, 0.8)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('不同融合模型在各指标上的表现', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_radar_chart.png'), dpi=300)
    
    plt.close('all')

def parse_args():
    parser = argparse.ArgumentParser(description='比较不同融合模型的性能')
    parser.add_argument('--models', nargs='+', default=['early', 'late', 'attention', 'transformer', 'hybrid'],
                        help='要比较的模型类型')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--results_dir', type=str, default=None, help='结果保存目录')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print(f"将比较以下融合模型: {', '.join(args.models)}")
    print(f"每个模型训练 {args.epochs} 轮")
    
    results_df = compare_models(model_types=args.models, num_epochs=args.epochs, results_dir=args.results_dir)
    
    # 打印最终结果表格
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\n最终比较结果:")
    print(results_df[['model_type', 'accuracy', 'precision', 'recall', 'f1', 'training_time', 'parameters']]) 