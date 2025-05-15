import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import streamlit as st

# Đọc dữ liệu
data = pd.read_csv("kf_coffee.csv")

# Chuẩn hóa tên cột
data.columns = data.columns.str.strip().str.lower()

# Hàm để tính tổng stock_decreased cho từng sản phẩm
def total_stock_decreased(history_str):
    try:
        history = json.loads(history_str).get("stock_history", [])
        return sum(float(entry.get("stock_decreased", 0)) for entry in history)
    except:
        return 0

# Áp dụng hàm
data["stock_decreased"] = data["stock_history"].apply(total_stock_decreased)

# Nhóm và lấy top 10 sản phẩm
top_products = data.groupby('name')['stock_decreased'].sum().sort_values(ascending=False).head(10)

# Vẽ biểu đồ bằng matplotlib
fig, ax = plt.subplots(figsize=(18, 7))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax)

# Thêm nhãn vào cột
for i, v in enumerate(top_products.values):
    ax.text(v + 5, i, str(int(v)), color='black', va='center', fontweight='bold')

ax.set_title("Top 10 sản phẩm bán chạy nhất")
ax.set_xlabel("Số lượng bán")
ax.set_ylabel("Sản phẩm")
plt.tight_layout()

# 👉 Hiển thị biểu đồ lên Streamlit
st.pyplot(fig)
