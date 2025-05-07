import sys
import traceback
sys.path.append('.')

import torch
import time
from multimodal.training.optimized_trainer import OptimizedMultiModalTrainer

def test_optimized_trainer():
    """测试优化的多模态训练器"""
    try:
        print("初始化优化训练器...")
        # 使用增强型注意力融合模型 - 这是整合了SE模块和残差连接的模型
        trainer = OptimizedMultiModalTrainer(model_type='enhanced_attention')
        
        print("准备数据...")
        train_loader, val_loader, test_loader = trainer.prepare_data(balance=True)
        
        # 训练模型
        print("\n开始训练过程...")
        # 设置较少的训练轮数进行快速测试
        num_epochs = 10  # 减少轮数，用于快速测试
        start_time = time.time()
        history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
        
        training_time = time.time() - start_time
        print(f"\n训练完成，用时: {training_time:.2f}秒")
        
        # 评估模型
        print("\n评估模型性能...")
        eval_results = trainer.evaluate(test_loader)
        print(f"测试集准确率: {eval_results['accuracy']:.4f}")
        print("分类报告:")
        for cls, metrics in eval_results['classification_report'].items():
            if isinstance(metrics, dict):
                print(f"  类别 {cls}: 精确率={metrics['precision']:.4f}, 召回率={metrics['recall']:.4f}, F1分数={metrics['f1-score']:.4f}")
        
        # 保存模型
        model_path = trainer.save_model()
        print(f"模型已保存到: {model_path}")
        
        print("\n测试完成！")
        return True
        
    except Exception as e:
        print("测试过程中发生错误:")
        print(str(e))
        traceback.print_exc()
        return False

def test_cross_validation():
    """测试交叉验证训练"""
    try:
        print("初始化训练器...")
        trainer = OptimizedMultiModalTrainer(model_type='enhanced_residual')
        
        # 创建数据集
        from multimodal.data.multimodal_dataset import MultiModalDataset
        
        # 设置数据文件路径
        hiseq_file = os.path.join('multimodal', 'data', 'processed', 'hiseq_processed.csv')
        rppa_file = os.path.join('multimodal', 'data', 'processed', 'rppa_processed.csv')
        
        print("准备数据集...")
        dataset = MultiModalDataset(
            hiseq_file=hiseq_file,
            rppa_file=rppa_file,
            balance=True
        )
        
        # 使用交叉验证训练
        print("\n开始交叉验证训练...")
        start_time = time.time()
        # 减少迭代次数用于快速测试
        trainer.training_params['epochs'] = 10
        trainer.training_params['patience'] = 5
        cv_results = trainer.train_with_cross_validation(dataset, n_splits=3)  # 减少折数
        
        cv_time = time.time() - start_time
        print(f"\n交叉验证完成，用时: {cv_time:.2f}秒")
        print(f"平均准确率: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        
        # 保存最佳模型
        model_path = trainer.save_model()
        print(f"最佳模型已保存到: {model_path}")
        
        print("\n测试完成！")
        return True
        
    except Exception as e:
        print("测试过程中发生错误:")
        print(str(e))
        traceback.print_exc()
        return False

def test_model_comparison():
    """测试比较不同模型"""
    try:
        print("比较不同模型架构的性能...")
        
        # 要测试的模型类型
        model_types = [
            'enhanced_attention',   # 增强型注意力融合
            'enhanced_residual',    # 增强型残差融合
            'enhanced_hybrid'       # 增强型混合融合
        ]
        
        results = {}
        
        for model_type in model_types:
            print(f"\n测试 {model_type} 模型...")
            
            # 初始化训练器
            trainer = OptimizedMultiModalTrainer(model_type=model_type)
            
            # 准备数据
            train_loader, val_loader, test_loader = trainer.prepare_data(balance=True)
            
            # 训练模型
            print(f"开始训练 {model_type} 模型...")
            start_time = time.time()
            history = trainer.train(train_loader, val_loader, num_epochs=10)  # 减少训练轮数用于快速比较
            
            training_time = time.time() - start_time
            
            # 评估模型
            eval_results = trainer.evaluate(test_loader)
            
            # 保存结果
            results[model_type] = {
                'accuracy': eval_results['accuracy'],
                'training_time': training_time,
                'history': history
            }
            
            print(f"{model_type} 模型测试完成，准确率: {eval_results['accuracy']:.4f}, 训练时间: {training_time:.2f}秒")
        
        # 输出比较结果
        print("\n模型对比结果:")
        print(f"{'模型类型':<20} {'准确率':<10} {'训练时间(秒)':<15}")
        print("-" * 45)
        for model_type, res in results.items():
            print(f"{model_type:<20} {res['accuracy']:.4f}    {res['training_time']:.2f}")
        
        # 找出最佳模型
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n最佳模型是: {best_model[0]}, 准确率: {best_model[1]['accuracy']:.4f}")
        
        print("\n测试完成！")
        return True
        
    except Exception as e:
        print("测试过程中发生错误:")
        print(str(e))
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("======= 测试优化的多模态模型 =======")
    # 执行单个模型训练测试
    test_optimized_trainer()
    
    # 如果需要，也可以执行下面的比较测试
    # print("\n\n")
    # test_cross_validation()
    # print("\n\n")
    # test_model_comparison() 