import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')  # 无GUI环境时使用
from config import DATA_PATHS, MODEL_PATHS, OUTPUT_DIRS, PREDICTION_PATH

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='生存模型训练、预测和分析工具')
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # train子命令
    train_parser = subparsers.add_parser('train', help='训练生存预测模型')
    train_parser.add_argument('-d', '--data', default=DATA_PATHS['default_with_survival'], 
                             help='训练数据文件路径')
    train_parser.add_argument('-o', '--output', default=OUTPUT_DIRS['final_model'],
                             help='模型输出目录')
    
    # predict子命令
    predict_parser = subparsers.add_parser('predict', help='使用已训练模型预测样本分组')
    predict_parser.add_argument('-r', '--rppa', default=DATA_PATHS['default_transposed'],
                               help='基因表达数据文件路径')
    predict_parser.add_argument('-m', '--model', default=MODEL_PATHS['ensemble'],
                               help='模型文件路径')
    predict_parser.add_argument('-o', '--output', default=PREDICTION_PATH,
                               help='预测结果输出文件')
    
    # analyze子命令
    analyze_parser = subparsers.add_parser('analyze', help='执行生存分析')
    analyze_parser.add_argument('-r', '--rppa', default=DATA_PATHS['default_transposed'],
                               help='基因表达数据文件路径')
    analyze_parser.add_argument('-c', '--clinical', default=DATA_PATHS['clinical'],
                               help='临床数据文件路径')
    analyze_parser.add_argument('-m', '--model', default=MODEL_PATHS['ensemble'],
                               help='模型文件路径')
    analyze_parser.add_argument('-o', '--output', default=OUTPUT_DIRS['survival_analysis'],
                               help='分析结果输出目录')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据命令执行相应功能
    if args.command == 'train':
        from train_model import train_survival_predictor
        print(f"开始训练模型，使用数据: {args.data}")
        train_survival_predictor(args.data, args.output)
        print(f"模型训练完成，保存在: {args.output}")
        
    elif args.command == 'predict':
        from predict import predict_from_rppa
        print(f"开始预测，使用基因表达数据: {args.rppa}")
        predict_from_rppa(args.rppa, args.model, args.output)
        print(f"预测完成，结果保存在: {args.output}")
        
    elif args.command == 'analyze':
        from survival_analysis import run_survival_analysis
        print(f"开始生存分析，使用基因表达数据: {args.rppa}")
        run_survival_analysis(args.rppa, args.clinical, args.model, args.output)
        print(f"生存分析完成，结果保存在: {args.output}")
        
    else:
        # 默认调用survival_analysis中的run_survival_analysis函数
        from survival_analysis import run_survival_analysis
        print("执行默认生存分析...")
        run_survival_analysis(
            DATA_PATHS['default_transposed'],
            DATA_PATHS['clinical'],
            MODEL_PATHS['ensemble'],
            OUTPUT_DIRS['survival_analysis']
        )
        print(f"生存分析已完成，结果保存在{OUTPUT_DIRS['survival_analysis']}目录下")

if __name__ == "__main__":
    main()
