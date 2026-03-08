#!/bin/bash

# 遍历当前目录下所有 .out 文件
for file in *.out; do
    # 检查是否存在匹配的文件（避免在没有任何 .out 文件时出错）
    if [ -f "$file" ]; then
        # 提取文件名（不带路径）
        filename=$(basename "$file")
        
        # 构建输出文件名
        output_file="opt-$filename"
        
        # 执行 wasm-opt 命令
        echo "正在优化: $filename -> $output_file"
        wasm-opt "$filename" --spill-pointers -o "$output_file"
        
        # 检查命令是否执行成功
        if [ $? -eq 0 ]; then
            echo "成功优化: $filename"
        else
            echo "错误: 优化 $filename 失败"
        fi
    fi
done

# 如果没有找到 .out 文件
if [ $(ls *.out 2>/dev/null | wc -l) -eq 0 ]; then
    echo "提示: 当前目录下没有找到 .out 文件"
fi
