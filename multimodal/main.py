"""
多模态训练系统主程序

该程序是多模态训练系统的入口点，提供命令行界面用于训练和评估融合模型。
可以通过设置不同的参数来控制训练过程，如模型类型、训练轮数、批量大小等。
"""

import sys
import os

# 将项目根目录添加到系统路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

import argparse
from multimodal.training.multimodal_trainer import MultiModalTrainer
from multimodal.utils.config import CONFIG


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模态融合模型训练和评估工具')
    
    # 模型设置
    parser.add_argument('--model_type', type=str, default='attention',
                        choices=['early', 'late', 'attention', 'transformer', 'hybrid'],
                        help='融合模型类型')
    
    # 训练设置
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数，默认使用配置文件中的设置')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批量大小，默认使用配置文件中的设置')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率，默认使用配置文件中的设置')
    
    # 执行模式
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'compare'],
                        help='执行模式：训练、评估或比较模型')
    
    return parser.parse_args()


def update_config(args):
    """根据命令行参数更新配置"""
    config = CONFIG.copy()
    
    if args.batch_size is not None:
        config['TRAINING_PARAMS']['batch_size'] = args.batch_size
    
    if args.lr is not None:
        config['TRAINING_PARAMS']['initial_lr'] = args.lr
    
    return config


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 更新配置
    config = update_config(args)
    
    # 创建训练器
    trainer = MultiModalTrainer(model_type=args.model_type, config=config)
    
    # 根据模式执行相应操作
    if args.mode == 'train':
        # 准备数据
        train_loader, val_loader = trainer.prepare_data()
        
        # 训练模型
        trainer.train(train_loader, val_loader, num_epochs=args.epochs)
        
        # 评估模型
        trainer.evaluate()
        
    elif args.mode == 'evaluate':
        # 评估已训练的模型
        trainer.prepare_data()  # 需要先准备数据来创建模型
        trainer.evaluate()
        
    elif args.mode == 'compare':
        # 比较不同模型的性能
        trainer.compare_models()
    
    print(f"{args.mode.capitalize()} 模式完成!")


if __name__ == "__main__":
    main() 