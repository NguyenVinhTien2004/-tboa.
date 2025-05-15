import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

# Tùy chỉnh layout
st.set_page_config(layout="wide")
st.title("📊 KPI Dashboard Doanh Số Cà Phê")

# Đọc dữ liệu từ file Excel
df = pd.read_excel("kf_coffee (1).xlsx", sheet_name="Trang tính1")

# Hàm tính số lượng bán trong khoảng ngày
def sum_stock_decreased_in_range(stock_history_str, start_date, end_date):
    try:
        history = json.loads(stock_history_str).get("stock_history", [])
        return sum(float(entry.get("stock_decreased", 0))
                   for entry in history if start_date <= entry["date"] <= end_date)
    except:
        return 0

# Tính số lượng bán theo các giai đoạn
df["Tuan_1"] = df["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-07", "2025-03-14"))
df["Tuan_2"] = df["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-15", "2025-03-22"))
df["Tuan_3"] = df["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-23", "2025-03-28"))
df["Ca_thang"] = df["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-07", "2025-03-28"))

# Tên các giai đoạn
period_label = {
    "Tuan_1": "Tuần 1 (07/03 → 14/03)",
    "Tuan_2": "Tuần 2 (15/03 → 22/03)",
    "Tuan_3": "Tuần 3 (23/03 → 28/03)",
    "Ca_thang": "Cả tháng (07/03 → 28/03)"
}

# Thứ tự giai đoạn để tính tăng trưởng
period_order = ["Tuan_1", "Tuan_2", "Tuan_3", "Ca_thang"]

# Lựa chọn giai đoạn từ dropdown
selected_period = st.selectbox("Chọn kỳ:", options=list(period_label.keys()), format_func=lambda x: period_label[x])

# Tính KPI
df["doanh_thu"] = df[selected_period] * df["price"]
total_units_sold = df[selected_period].sum()
total_revenue = df["doanh_thu"].sum()
top_product = df.sort_values(by=selected_period, ascending=False).iloc[0]["name"]

# Tính tăng trưởng so với kỳ trước
current_idx = period_order.index(selected_period)
if current_idx > 0:
    prev_period = period_order[current_idx - 1]
    prev_units_sold = df[prev_period].sum()
    growth = ((total_units_sold - prev_units_sold) / prev_units_sold * 100) if prev_units_sold > 0 else 0
    growth_text = f"{growth:+.1f}%"
else:
    growth_text = "N/A"

growth_title = "Tăng trưởng so với tuần trước" if selected_period != "Ca_thang" else "Tăng trưởng của cả tháng"

# Hiển thị KPI cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Tổng doanh thu", f"{int(total_revenue):,} VNĐ")
col2.metric("📦 Tổng sản phẩm bán ra", f"{int(total_units_sold):,} sản phẩm")
col3.metric("🔥 Sản phẩm bán chạy nhất", top_product)
col4.metric("📊 " + growth_title, growth_text)

# Biểu đồ cột sản phẩm theo doanh thu kỳ được chọn
st.subheader(f"Biểu đồ doanh thu theo sản phẩm – {period_label[selected_period]}")
top_products = df.sort_values(by="doanh_thu", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(top_products["name"], top_products["doanh_thu"], color="#4CAF50")
ax.set_title("Top 10 sản phẩm doanh thu cao", fontsize=14)
ax.set_ylabel("Doanh thu (VNĐ)")
ax.set_xticklabels(top_products["name"], rotation=45, ha='right')
st.pyplot(fig)
