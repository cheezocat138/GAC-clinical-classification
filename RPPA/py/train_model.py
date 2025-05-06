import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# 导入自定义模块和配置
from config import *
from models import HybridSurvivalPredictor, WeightedCrossEntropyLoss
from utils import RPPADataset, clip_gradients, cyclical_lr, enhanced_feature_selection, add_cluster_features, ensure_dir, plot_confusion_matrix

def train_survival_predictor(data_path=DATA_PATHS['rppa_with_survival'], output_dir=OUTPUT_DIRS['final_model']):
    """训练生存预测模型"""
    # 设置随机种子，确保结果可复现
    torch.manual_seed(TRAINING_PARAMS['random_seed'])
    np.random.seed(TRAINING_PARAMS['random_seed'])
    
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 加载数据
    df = pd.read_csv(data_path)
    
    # 去除survival_group_code为空的行
    df = df.dropna(subset=['survival_group_code'])
    
    # 数据预处理 - 特征工程
    # 处理缺失值 - 使用中位数填充
    df_features = df.drop(columns=['survival_group_code', 'sample'])
    df_features = df_features.fillna(df_features.median())
    
    # 分离特征和目标变量
    X = df_features.values
    y = df['survival_group_code'].values - 1  # 将类别转换为0, 1, 2
    
    # 打印类别分布
    unique_classes, counts = np.unique(y, return_counts=True)
    print("类别分布:", dict(zip(unique_classes, counts)))
    
    # 应用增强型特征选择
    X_selected, selected_indices = enhanced_feature_selection(X, y)
    selected_feature_names = df_features.columns[selected_indices].tolist()
    
    print(f"原始特征数: {X.shape[1]}, 选择后特征数: {X_selected.shape[1]}")
    
    # 应用聚类特征增强
    
    # 创建特征处理管道
    feature_pipeline = Pipeline([
        ('feature_selection', FunctionTransformer(lambda x: x[:, selected_indices])),
        ('add_clusters', FunctionTransformer(lambda x: add_cluster_features(x, n_clusters=CLUSTER_PARAMS['n_clusters'])))
    ])
    
    # 应用特征管道
    X_enhanced = feature_pipeline.fit_transform(X)
    print(f"增强后特征数: {X_enhanced.shape[1]}")
    
    # 使用SMOTE处理类别不平衡
    print("应用SMOTE平衡类别分布...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_enhanced, y)
    print(f"SMOTE前数据规模: {X_enhanced.shape}, 应用后: {X_resampled.shape}")
    print("平衡后类别分布:", dict(zip(*np.unique(y_resampled, return_counts=True))))
    
    # 计算类别权重，对平衡后的数据可以使用均等权重
    class_weights = torch.FloatTensor(TRAINING_PARAMS['class_weights'])
    
    # 数据分割 - 使用分层K折交叉验证
    skf = StratifiedKFold(n_splits=TRAINING_PARAMS['n_splits'], shuffle=True, random_state=TRAINING_PARAMS['random_seed'])
    fold_accuracy = []
    
    # 数据标准化
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    
    # 检查是否有无穷或NaN值
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 用于K折交叉验证的结果字典
    cv_results = {
        'train_losses': [],
        'val_losses': [],
        'val_accs': [],
        'best_val_accs': [],
        'fold_models': [],
        'fold_predictions': [],
        'fold_labels': [],
        'fold_probabilities': []
    }
    
    # 开始交叉验证
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_resampled)):
        print(f"\n开始训练第 {fold+1} 折...")
        
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # 创建数据加载器
        train_dataset = RPPADataset(X_train_tensor, y_train_tensor)
        test_dataset = RPPADataset(X_test_tensor, y_test_tensor)
        
        # 动态调整批量大小
        batch_size = min(TRAINING_PARAMS['batch_size'], len(train_dataset) // 10)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 初始化简化后的模型
        input_size = X_train.shape[1]
        model = HybridSurvivalPredictor(input_size)
        
        # 定义损失函数和优化器
        criterion = WeightedCrossEntropyLoss(weight=class_weights)
        
        # 使用SGD优化器代替AdamW，有时能带来更好的泛化性能
        initial_lr = TRAINING_PARAMS['initial_lr']
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.001)
        
        # 获取每个训练周期的步数
        steps_per_epoch = len(train_loader)
        
        # 使用修正的循环退火学习率调度器
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=cyclical_lr(
                initial_lr, 
                max_lr=TRAINING_PARAMS['max_lr'], 
                min_lr=TRAINING_PARAMS['min_lr'], 
                stepsize=5 * steps_per_epoch
            )
        )
        
        # 保存最佳模型
        def save_checkpoint(state, filename=os.path.join(output_dir, FILE_FORMATS['model_fold'].format(fold))):
            torch.save(state, filename)
        
        # 训练模型 - 添加早停和混合精度训练
        def train_epoch(model, train_loader, criterion, optimizer, scheduler):
            model.train()
            running_loss = 0.0
            
            # 使用混合精度训练加速
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            
            for inputs, labels in train_loader:
                # 使用混合精度训练
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    # 反向传播和优化
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_gradients(model, clip_value=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 标准训练
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    clip_gradients(model, clip_value=1.0)
                    optimizer.step()
                
                running_loss += loss.item()
                scheduler.step()  # 每批次更新一次学习率
                
            return running_loss / len(train_loader)
        
        def validate_model(model, test_loader, criterion):
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(test_loader)
            val_acc = val_correct / val_total
            
            return val_loss, val_acc
        
        # 训练模型主循环
        train_losses = []
        val_losses = []
        val_accs = []
        best_val_loss = float('inf')
        best_val_acc = 0
        patience = TRAINING_PARAMS['patience']
        patience_counter = 0
        
        for epoch in range(TRAINING_PARAMS['max_epochs']):  # 最大训练周期
            # 训练一个epoch
            epoch_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler)
            train_losses.append(epoch_loss)
            
            # 验证
            val_loss, val_acc = validate_model(model, test_loader, criterion)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型(基于验证损失)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                })
            else:
                patience_counter += 1
                
            # 如果验证准确率更高，也保存模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, os.path.join(output_dir, FILE_FORMATS['model_fold_acc'].format(fold)))
            
            # 发现NaN值时提早停止
            if np.isnan(epoch_loss) or np.isnan(val_loss):
                print(f"发现NaN值，在第{epoch+1}轮停止训练")
                break
            
            # 早停
            if patience_counter >= patience:
                print(f"早停触发，验证损失{patience}轮未改善")
                break
                
            # 打印训练进度
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == TRAINING_PARAMS['max_epochs'] - 1:
                print(f'Epoch [{epoch+1}/150], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 加载最佳模型
        best_model_data = torch.load(
            os.path.join(output_dir, FILE_FORMATS['model_fold_acc'].format(fold)), 
            weights_only=False
        )
        model.load_state_dict(best_model_data['model_state_dict'])
        
        # 在测试集上评估模型
        def evaluate_model(model, test_loader):
            model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    
                    probs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_preds.extend(predicted.numpy())
                    all_labels.extend(labels.numpy())
                    all_probs.extend(probs.numpy())
            
            # 计算准确率
            accuracy = accuracy_score(all_labels, all_preds)
            
            return accuracy, all_preds, all_labels, all_probs

        # 评估最佳模型
        fold_acc, fold_preds, fold_labels, fold_probs = evaluate_model(model, test_loader)
        print(f"第 {fold+1} 折最佳验证准确率: {best_val_acc:.4f}, 测试准确率: {fold_acc:.4f}")
        
        # 保存结果
        cv_results['train_losses'].append(train_losses)
        cv_results['val_losses'].append(val_losses)
        cv_results['val_accs'].append(val_accs)
        cv_results['best_val_accs'].append(best_val_acc)
        cv_results['fold_models'].append(model.state_dict())
        cv_results['fold_predictions'].append(fold_preds)
        cv_results['fold_labels'].append(fold_labels)
        cv_results['fold_probabilities'].append(fold_probs)
        
        # 打印分类报告
        print("\n分类报告 (第 {} 折):".format(fold+1))
        print(classification_report(fold_labels, fold_preds, target_names=['Group 1', 'Group 2', 'Group 3'], zero_division=0))
        
        fold_accuracy.append(fold_acc)

    # 输出交叉验证结果
    print("\n交叉验证结果:")
    print(f"平均准确率: {np.mean(fold_accuracy):.4f} ± {np.std(fold_accuracy):.4f}")
    print(f"各折准确率: {fold_accuracy}")
    print(f"最佳折验证准确率: {max(cv_results['best_val_accs']):.4f}")

    # 集成预测
    def ensemble_predictions(models, X_test_tensor):
        all_predictions = []
        all_probabilities = []
        
        # 收集所有折模型的预测
        for model_state in models:
            model = HybridSurvivalPredictor(X_test_tensor.shape[1])
            model.load_state_dict(model_state)
            model.eval()
            
            with torch.no_grad():
                outputs = model(X_test_tensor)
                
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_predictions.append(preds.numpy())
                all_probabilities.append(probs.numpy())
        
        # 投票法确定最终预测
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # 使用软投票(概率平均)
        avg_probabilities = np.mean(all_probabilities, axis=0)
        final_predictions = np.argmax(avg_probabilities, axis=1)
        
        return final_predictions, avg_probabilities

    # 创建测试集的完整张量
    X_test_full = torch.FloatTensor(X_scaled)
    y_test_full = torch.LongTensor(y_resampled)

    # 执行集成预测
    ensemble_preds, ensemble_probs = ensemble_predictions(cv_results['fold_models'], X_test_full)

    # 评估集成模型
    ensemble_acc = accuracy_score(y_resampled, ensemble_preds)
    print(f"\n集成模型准确率: {ensemble_acc:.4f}")
    print("\n集成模型分类报告:")
    print(classification_report(y_resampled, ensemble_preds, target_names=['Group 1', 'Group 2', 'Group 3'], zero_division=0))

    # 绘制混淆矩阵
    plot_confusion_matrix(
        y_resampled, ensemble_preds, 
        classes=PLOT_PARAMS['group_names'], 
        output_path=os.path.join(output_dir, FILE_FORMATS['confusion_matrix']), 
        title="Ensemble Model Confusion Matrix"
    )

    # 选择最佳模型作为最终模型
    best_fold_idx = np.argmax(cv_results['best_val_accs'])
    print(f"选择第 {best_fold_idx+1} 折模型作为最终单一模型")

    # 保存最终单一模型
    final_model = HybridSurvivalPredictor(X_scaled.shape[1])
    final_model.load_state_dict(cv_results['fold_models'][best_fold_idx])
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'scaler': scaler,
        'selected_features': selected_feature_names,
        'selected_indices': selected_indices,
        'feature_pipeline': feature_pipeline,  # 保存特征处理管道
        'input_shape': X_scaled.shape[1],
        'original_shape': X.shape[1],
        'selected_shape': X_selected.shape[1],
        'enhanced_shape': X_enhanced.shape[1],
        'accuracy': fold_accuracy[best_fold_idx],
        'cv_accuracy_mean': np.mean(fold_accuracy),
        'cv_accuracy_std': np.std(fold_accuracy)
    }, MODEL_PATHS['final'])

    # 保存集成模型结果
    torch.save({
        'fold_models': cv_results['fold_models'],
        'ensemble_accuracy': ensemble_acc,
        'scaler': scaler,
        'selected_features': selected_feature_names,
        'selected_indices': selected_indices,
        'feature_pipeline': feature_pipeline,  # 保存特征处理管道
        'input_shape': X_scaled.shape[1],
        'original_shape': X.shape[1],
        'selected_shape': X_selected.shape[1],
        'enhanced_shape': X_enhanced.shape[1],
        'cv_accuracy_mean': np.mean(fold_accuracy),
        'cv_accuracy_std': np.std(fold_accuracy)
    }, MODEL_PATHS['ensemble'])

    # 可视化训练历史和交叉验证结果
    plt.figure(figsize=PLOT_PARAMS['figsize_large'])

    # 所有折的训练损失曲线
    plt.subplot(2, 2, 1)
    for i, losses in enumerate(cv_results['train_losses']):
        plt.plot(losses, label=f'Fold {i+1}')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 所有折的验证准确率曲线
    plt.subplot(2, 2, 2)
    for i, accs in enumerate(cv_results['val_accs']):
        plt.plot(accs, label=f'Fold {i+1}')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 各折最佳准确率比较
    plt.subplot(2, 2, 3)
    plt.bar(range(1, len(fold_accuracy)+1), fold_accuracy)
    plt.axhline(y=np.mean(fold_accuracy), color='r', linestyle='-', label=f'Mean: {np.mean(fold_accuracy):.4f}')
    plt.axhline(y=ensemble_acc, color='g', linestyle='--', label=f'Ensemble: {ensemble_acc:.4f}')
    plt.title('Best Accuracy by Fold and Ensemble')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, len(fold_accuracy)+1))
    plt.legend()

    # 混淆矩阵
    plt.subplot(2, 2, 4)
    cm = confusion_matrix(y_resampled, ensemble_preds)
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Ensemble Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1, 2], ['Group 1', 'Group 2', 'Group 3'])
    plt.yticks([0, 1, 2], ['Group 1', 'Group 2', 'Group 3'])

    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], 
                   ha="center", va="center", color="white" if cm[i, j] > 10 else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, FILE_FORMATS['cv_results']), dpi=PLOT_PARAMS['dpi'])
    plt.close()

    # 绘制特征重要性图
    plt.figure(figsize=PLOT_PARAMS['figsize_medium'])

    # 使用X_resampled而不是X_enhanced来训练随机森林
    # 注意：只使用SMOTE后的原始特征部分，不包括聚类特征
    X_original_features = X_resampled[:, :X_selected.shape[1]]  # 只取原始特征部分

    rf_final = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_final.fit(X_original_features, y_resampled)
    importances = rf_final.feature_importances_
    sorted_idx = np.argsort(importances)[-25:]
    top_features = [selected_feature_names[i] for i in sorted_idx]
    top_scores = importances[sorted_idx]

    plt.barh(range(len(top_features)), top_scores, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.title('Top 25 Feature Importance')
    plt.xlabel('Random Forest Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, FILE_FORMATS['feature_importance']), dpi=PLOT_PARAMS['dpi'])
    plt.close()

    print("训练完成，最终模型和集成模型已保存")

    # 返回训练的结果信息
    return {
        'scaler': scaler,
        'selected_features': selected_feature_names,
        'selected_indices': selected_indices,
        'best_fold_idx': best_fold_idx,
        'ensemble_acc': ensemble_acc,
        'fold_accuracy': fold_accuracy,
        'model_paths': {
            'final_model': f'{output_dir}/final_survival_predictor_model.pth',
            'ensemble_model': f'{output_dir}/ensemble_survival_predictor_model.pth'
        }
    }

if __name__ == "__main__":
    # 直接调用训练函数执行模型训练
    train_results = train_survival_predictor()
    print("模型训练完成。")
