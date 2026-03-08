#!/bin/bash

# 设置超时时间（秒）
TIMEOUT=20

echo "========================================"
echo "开始运行 opt-*.out 文件测试"
echo "超时限制: ${TIMEOUT}秒"
echo "========================================"
echo ""

# 统计信息
total_files=0
success_count=0
timeout_count=0
error_count=0

# 遍历当前目录下所有 *.out 文件
for file in *.out; do
    # 检查是否存在匹配的文件
    if [ -f "$file" ]; then
        # 更新统计
        ((total_files++))
        
        # 提取文件名（不带路径）
        filename=$(basename "$file")
        
        # 记录开始时间
        start_time=$(date +%s)
        
        # 打印文件名
        echo "=========================================="
        echo "文件 $total_files: $filename"
        echo "开始时间: $(date '+%H:%M:%S')"
        echo "=========================================="
        
        # 使用 timeout 命令运行 wasmtime，设置超时时间
        timeout ${TIMEOUT}s wasmtime "$filename"
        
        # 检查命令执行结果
        result=$?
        
        # 记录结束时间
        end_time=$(date +%s)
        elapsed_time=$((end_time - start_time))
        
        # 根据退出码判断执行情况
        if [ $result -eq 0 ]; then
            echo ""
            echo "✅ 成功运行: $filename"
            echo "   运行时间: ${elapsed_time}秒"
            ((success_count++))
        elif [ $result -eq 124 ]; then
            echo ""
            echo "⏰ 超时: $filename 运行时间超过 ${TIMEOUT} 秒"
            echo "❌ 运行失败: $filename (超时终止)"
            ((timeout_count++))
        else
            echo ""
            echo "❌ 运行失败: $filename (退出码: $result)"
            echo "   运行时间: ${elapsed_time}秒"
            ((error_count++))
        fi
        
        # 添加空行分隔不同文件的输出
        echo ""
        echo ""
    fi
done

# 输出统计信息
echo "========================================"
echo "测试完成 - 统计信息"
echo "========================================"
echo "总文件数: $total_files"
echo "成功: $success_count"
echo "超时: $timeout_count"
echo "失败: $error_count"
echo "========================================"

# 如果没有找到 opt-*.out 文件
if [ $total_files -eq 0 ]; then
    echo "提示: 当前目录下没有找到 opt-*.out 文件"
    echo "请先运行优化脚本生成 opt-*.out 文件"
fi
