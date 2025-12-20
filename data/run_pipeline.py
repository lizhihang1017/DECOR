import os
import sys
import time
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor


def run_script(script_path):
    """执行指定的Python脚本并实时显示输出"""
    if not os.path.exists(script_path):
        print(f"错误: 文件 '{script_path}' 不存在!")
        return False

    print(f"正在执行: {script_path}")
    start_time = time.time()

    try:
        # 创建子进程并捕获输出
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # 行缓冲
            universal_newlines=True
        )

        # 创建线程实时读取输出
        def read_output():
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                # 添加前缀并实时打印
                print(f"[{os.path.basename(script_path)}] {line}", end='')

        # 启动输出读取线程
        output_thread = threading.Thread(target=read_output)
        output_thread.daemon = True  # 设置为守护线程
        output_thread.start()

        # 等待进程完成
        process.wait()

        # 检查退出状态
        if process.returncode == 0:
            duration = time.time() - start_time
            print(f"✓ 成功执行 {script_path} (耗时: {duration:.2f}秒)")
            return True
        else:
            duration = time.time() - start_time
            print(f"✗ 执行失败 {script_path} (耗时: {duration:.2f}秒, 退出码: {process.returncode})")
            return False

    except Exception as e:
        duration = time.time() - start_time
        print(f"✗ 执行异常 {script_path} (耗时: {duration:.2f}秒)")
        print(f"异常信息: {str(e)}")
        return False


def run_scripts_concurrently(script_list):
    """并行执行多个脚本并等待所有完成"""
    print(f"\n{'=' * 30} 开始并行执行 {'=' * 30}")
    print(f"并行执行脚本: {', '.join([os.path.basename(s) for s in script_list])}")

    with ThreadPoolExecutor(max_workers=len(script_list)) as executor:
        # 提交所有任务
        futures = {executor.submit(run_script, script): script for script in script_list}

        # 等待所有任务完成
        results = {}
        for future in futures:
            script = futures[future]
            try:
                results[script] = future.result()
            except Exception as e:
                print(f"✗ 并行执行异常: {script} - {str(e)}")
                results[script] = False

    print(f"{'=' * 30} 并行执行完成 {'=' * 30}\n")

    # 检查所有脚本是否都成功
    return all(results.values())


def main():
    # 定义要顺序执行的脚本
    sequential_scripts = [
        # "step1_filter_llm_knowledge.py",
        # "step2_retrival_splade.py"
    ]

    # 定义要并行执行的脚本
    parallel_scripts = [
        # "step3_propositional_context_1.py",
        # "step3_propositional_context_2.py",
        # "step3_propositional_context_3.py",
        "step3_propositional_context_4.py",
        "step3_propositional_context_5.py",
        "step3_propositional_context_6.py"
    ]

    print("=" * 60)
    print("开始执行处理流程".center(60))
    print("=" * 60)
    print("说明: 每个脚本的输出将以 [脚本名] 为前缀实时显示\n")

    # 顺序执行每个脚本
    for script in sequential_scripts:
        success = run_script(script)
        if not success:
            print("\n" + "!" * 60)
            print(f"处理流程在 {script} 处中断".center(60))
            print("!" * 60)
            sys.exit(1)  # 退出并返回错误代码
        print()  # 添加空行分隔不同脚本的输出

    # 并行执行step3脚本
    all_success = run_scripts_concurrently(parallel_scripts)

    if not all_success:
        print("\n" + "!" * 60)
        print("并行执行中有脚本失败!".center(60))
        print("!" * 60)
        sys.exit(1)  # 退出并返回错误代码

    print("\n" + "=" * 60)
    print("所有处理步骤成功完成!".center(60))
    print("=" * 60)


if __name__ == "__main__":
    main()