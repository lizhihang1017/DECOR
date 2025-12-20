import pandas as pd
import eval_utils
from tqdm import tqdm
import string
import argparse
import sys


def normalize_text(text: str) -> str:
    """标准化文本：小写、去标点、去空格"""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())


def superset_metric(gold: str, generated: str) -> bool:
    """检查黄金答案是否为生成答案的子集（标准化后）"""
    return normalize_text(gold) in normalize_text(generated)


if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='评估模型生成结果')
    parser.add_argument('--input_files', nargs='+', required=True,
                        help='要评估的输入文件列表，用空格分隔多个文件')
    parser.add_argument('--output_summary', type=str, default=None,
                        help='汇总结果的输出文件路径（可选）')
    args = parser.parse_args()

    # 获取输入文件列表
    input_file_list = args.input_files

    # 存储所有评估结果
    all_results = []

    for input_file in input_file_list:
        # 读取JSON文件
        try:
            input_df = pd.read_json(input_file, orient='records')
        except Exception as e:
            print(f"错误读取文件 {input_file}: {e}")
            continue

        # 直接使用gold_answers列（假设已经是list类型）
        input_df['em'] = input_df.apply(lambda row: eval_utils.single_ans_em(
            gold=row['gold_answers'],
            pred=row['generated_answer']
        ), axis=1)

        input_df['f1'] = input_df.apply(lambda row: eval_utils.single_ans_f1(
            gold=row['gold_answers'],
            pred=row['generated_answer']
        ), axis=1)

        # 计算Superset指标（带进度条）
        tqdm.pandas(desc=f"计算Superset指标 - {input_file}")
        input_df['superset'] = input_df.progress_apply(
            lambda row: superset_metric(row['gold_answers'], row['generated_answer']),
            axis=1
        )

        # 打印当前文件结果
        filename = input_file.split("/")[-1]
        em_score = input_df['em'].mean()
        f1_score = input_df['f1'].mean()
        superset_score = input_df['superset'].mean()

        print("\n=== {file} 评估结果 ===".format(file=filename))
        print(f"平均EM分数: {em_score:.4f}")
        print(f"平均F1分数: {f1_score:.4f}")
        print(f"平均Superset分数: {superset_score:.4f}")

        # 保存当前文件结果
        all_results.append({
            "filename": filename,
            "em": em_score,
            "f1": f1_score,
            "superset": superset_score,
            "total_samples": len(input_df),
            "file_path": input_file
        })

        # 保存带指标的完整数据（可选）
        output_file = input_file.replace(".json", "_with_metrics.json")
        try:
            input_df.to_json(output_file, orient='records', indent=2)
            print(f"带指标的结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存结果文件失败: {e}")

    # 输出汇总结果
    print("\n=== 汇总评估结果 ===")
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))

    # 保存汇总结果到文件（如果指定了输出文件）
    if args.output_summary:
        try:
            summary_df.to_csv(args.output_summary, index=False)
            print(f"\n汇总结果已保存到: {args.output_summary}")
        except Exception as e:
            print(f"保存汇总结果失败: {e}")