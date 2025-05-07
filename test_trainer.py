import sys
import traceback
sys.path.append('.')

import torch
import time
from multimodal.training.multimodal_trainer import MultiModalTrainer

def test_trainer():
    """测试多模态训练器"""
    try:
        print("初始化训练器...")
        # 使用最复杂的Transformer融合模型
        trainer = MultiModalTrainer(model_type='transformer')
        
        # 修改早停设置
        trainer.training_params['patience'] = 10  # 减少早停轮数，加快训练
        
        print("准备数据...")
        train_loader, val_loader = trainer.prepare_data()
        
        # 训练模型足够的轮次，确保训练充分但避免过长时间
        print("\n开始训练过程...")
        num_epochs = 100  # 设置适当的轮数
        start_time = time.time()
        history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
        
        training_time = time.time() - start_time
        print(f"\n训练完成，用时: {training_time:.2f}秒")
        
        # 评估模型
        print("\n评估模型性能...")
        eval_results = trainer.evaluate(val_loader)
        
        # 打印结果
        print(f"\n评估结果:")
        print(f"准确率: {eval_results['accuracy']:.4f}")
        
        # 查看最佳验证准确率和对应的轮次
        best_val_acc = trainer.history['best_val_acc']
        best_epoch = trainer.history['best_epoch']
        print(f"最佳验证准确率: {best_val_acc:.4f} (轮次 {best_epoch})")
        
        print("\n训练器测试完成!")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试多模态训练器...")
    success = test_trainer()
    print(f"测试{'成功' if success else '失败'}!") 