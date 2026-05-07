"""
数据预处理脚本
功能：清洗、转换、特征工程
运行方式：python 03_scripts/01_data_preprocess.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("电商消费者数据预处理脚本")
print("=" * 100)

# ==================================================
# 第1部分：加载数据
# ==================================================
print("\n【步骤1】加载原始数据...")
try:
    df = pd.read_csv('01_data/raw/shopping_trends.csv')
    print("✅ 数据加载成功")
    print(f"   数据形状: {df.shape}")
    print(f"   消费者数: {df['Customer ID'].nunique()}")
except FileNotFoundError:
    print("❌ 错误：找不到文件 01_data/raw/shopping_trends.csv")
    print("   请确认文件路径是否正确")
    exit(1)

# ==================================================
# 第2部分：数据清洗
# ==================================================
print("\n【步骤2】数据清洗...")

# 1. 缺失值检查与处理
print("  缺失值检查:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "  无缺失值")

# 2. 重复值检查与处理
print("  重复值检查:")
duplicates = df.duplicated().sum()
if duplicates > 0:
    print(f"  发现 {duplicates} 条重复记录，已删除")
    df = df.drop_duplicates()
else:
    print("  无重复记录")

# 3. 异常值处理（以消费金额为例）
print("  消费金额异常值处理:")
Q1 = df['Purchase Amount (USD)'].quantile(0.25)
Q3 = df['Purchase Amount (USD)'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['Purchase Amount (USD)'] < lower) | (df['Purchase Amount (USD)'] > upper)]
if len(outliers) > 0:
    print(f"  发现 {len(outliers)} 条异常记录，已保留（仅用于后续分析）")
else:
    print("  无异常记录")

# ==================================================
# 第3部分：特征工程
# ==================================================
print("\n【步骤3】特征工程...")

# 1. 购买频率编码
freq_map = {
    'Fortnightly': 1,
    'Weekly': 2,
    'Monthly': 3,
    'Quarterly': 4,
    'Annually': 5
}
df['Frequency_Encoded'] = df['Frequency of Purchases'].map(freq_map)

# 2. 消费金额分组
df['Spending_Level'] = pd.cut(
    df['Purchase Amount (USD)'],
    bins=[0, 50, 100, 150, np.inf],
    labels=['低消费', '中消费', '高消费', '超高消费']
)

print("  新增特征: Frequency_Encoded, Spending_Level")

# ==================================================
# 第4部分：保存处理后的数据
# ==================================================
print("\n【步骤4】保存处理后的数据...")
df.to_csv('01_data/processed/shopping_trends_processed.csv', index=False)
print("✅ 处理后数据已保存至: 01_data/processed/shopping_trends_processed.csv")
print(f"   最终数据形状: {df.shape}")

print("\n" + "=" * 100)
print("数据预处理完成！")
print("=" * 100)