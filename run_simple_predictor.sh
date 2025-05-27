#!/bin/bash

# 导航到项目目录
cd "$(dirname "$0")"

# 激活虚拟环境
source venv/bin/activate

# 默认值
TICKER="NVDA"
DAYS=30
YEARS=2

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--ticker)
      TICKER="$2"
      shift 2
      ;;
    -d|--days)
      DAYS="$2"
      shift 2
      ;;
    -y|--years)
      YEARS="$2"
      shift 2
      ;;
    *)
      echo "未知选项: $1"
      exit 1
      ;;
  esac
done

echo "====================================================="
echo "       简化版股票预测系统 - 不依赖Darts库"
echo "====================================================="
echo "分析股票: $TICKER"
echo "预测天数: $DAYS 天"
echo "历史数据: $YEARS 年"
echo "-----------------------------------------------------"

# 运行预测脚本
python src/simple_predictor.py "$TICKER" --days "$DAYS" --years "$YEARS"

echo "-----------------------------------------------------"
echo "预测完成!"
echo "结果保存在: data/${TICKER}_prediction.png"
echo "=====================================================" 