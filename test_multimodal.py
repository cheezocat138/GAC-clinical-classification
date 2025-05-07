import sys
import traceback
sys.path.append('.')

try:
    print("开始导入模块...")
    from multimodal.utils.config import CONFIG
    print("成功导入CONFIG")
    from multimodal.data.rppa_data import RPPADataProcessor
    print("成功导入RPPADataProcessor")
    from multimodal.data.hiseq_data import HiSeqDataProcessor
    print("成功导入HiSeqDataProcessor")
    from multimodal.data.multimodal_dataset import MultiModalDataProcessor
    print("成功导入MultiModalDataProcessor")
except Exception as e:
    print(f"导入模块失败: {e}")
    traceback.print_exc()
    sys.exit(1)

def test_sample_matching():
    """测试样本匹配逻辑"""
    import pandas as pd
    import numpy as np
    
    print("加载原始数据文件以检查样本ID...")
    try:
        hiseq = pd.read_csv('multimodal/data/raw/hiseq/HiSeqV2_transposed.csv', index_col=0)
        rppa = pd.read_csv('multimodal/data/raw/rppa/RPPA_transposed.csv')
        survival = pd.read_csv('multimodal/data/raw/common/survival_three_groups.csv')
        
        print(f"HiSeq数据形状: {hiseq.shape}")
        print(f"RPPA数据形状: {rppa.shape}")
        print(f"生存数据形状: {survival.shape}")
        
        print("\nHiSeq样本ID类型:", type(hiseq.index))
        print("HiSeq样本ID示例(前5个):", list(hiseq.index)[:5])
        
        print("\nRPPA列名:", list(rppa.columns))
        print("RPPA样本ID示例(前5个):", list(rppa['sample'])[:5] if 'sample' in rppa.columns else "找不到sample列")
        
        hiseq_ids = set(hiseq.index)
        rppa_ids = set(rppa['sample']) if 'sample' in rppa.columns else set()
        common = hiseq_ids.intersection(rppa_ids)
        
        print(f"\n直接匹配结果:")
        print(f"HiSeq样本数量: {len(hiseq_ids)}")
        print(f"RPPA样本数量: {len(rppa_ids)}")
        print(f"共同样本数量: {len(common)}")
        print("共同样本示例(前5个):", list(common)[:5] if common else "无共同样本")
        
        # 尝试调整样本ID并重新匹配
        print("\n尝试去除样本ID中的后缀...")
        hiseq_ids_simple = {id.split('-')[0] + '-' + id.split('-')[1] + '-' + id.split('-')[2] for id in hiseq_ids}
        rppa_ids_simple = {id.split('-')[0] + '-' + id.split('-')[1] + '-' + id.split('-')[2] for id in rppa_ids}
        common_simple = hiseq_ids_simple.intersection(rppa_ids_simple)
        
        print(f"简化后HiSeq样本数量: {len(hiseq_ids_simple)}")
        print(f"简化后RPPA样本数量: {len(rppa_ids_simple)}")
        print(f"简化后共同样本数量: {len(common_simple)}")
        print("简化后共同样本示例(前5个):", list(common_simple)[:5] if common_simple else "无共同样本")
        
    except Exception as e:
        print(f"样本匹配测试失败: {e}")
        traceback.print_exc()

def test_multimodal_dataset():
    try:
        print("初始化多模态数据处理器...")
        processor = MultiModalDataProcessor()
        print("处理器初始化成功")
        
        print("\n开始加载数据...")
        try:
            hiseq_features, rppa_features, labels = processor.load_data()
            print(f"数据加载完成。HiSeq特征形状: {hiseq_features.shape}, RPPA特征形状: {rppa_features.shape}, 标签形状: {labels.shape}")
        except Exception as e:
            print(f"数据加载失败: {e}")
            traceback.print_exc()
            return
        
        print("\n开始预处理数据...")
        try:
            hiseq_processed, rppa_processed, labels_processed = processor.preprocess_data(hiseq_features, rppa_features, labels)
            print(f"数据预处理完成。HiSeq处理后形状: {hiseq_processed.shape}, RPPA处理后形状: {rppa_processed.shape}, 处理后标签形状: {labels_processed.shape}")
        except Exception as e:
            print(f"数据预处理失败: {e}")
            traceback.print_exc()
            return
        
        print("\n创建数据加载器...")
        try:
            train_loader, test_loader = processor.get_dataloaders(hiseq_processed, rppa_processed, labels_processed)
            print("数据加载器创建成功")
        except Exception as e:
            print(f"创建数据加载器失败: {e}")
            traceback.print_exc()
            return
        
        print("\n测试数据加载器...")
        try:
            for batch_idx, (hiseq_batch, rppa_batch, label_batch) in enumerate(train_loader):
                print(f"Batch {batch_idx}: HiSeq形状: {hiseq_batch.shape}, RPPA形状: {rppa_batch.shape}, 标签形状: {label_batch.shape}")
                if batch_idx >= 2:  # 只显示前3个批次
                    break
            print("数据加载器测试成功")
        except Exception as e:
            print(f"测试数据加载器失败: {e}")
            traceback.print_exc()
            return
        
        print("\n保存处理后的数据...")
        try:
            processor.save_processed_data()
            print("数据保存成功")
        except Exception as e:
            print(f"保存数据失败: {e}")
            traceback.print_exc()
            return
        
        print("\n测试完成!")
    except Exception as e:
        print(f"测试过程中出现未捕获的异常: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("脚本开始执行...")
    # 先测试样本匹配
    test_sample_matching()
    # 然后测试完整的数据处理流程
    test_multimodal_dataset()
    print("脚本执行结束") 