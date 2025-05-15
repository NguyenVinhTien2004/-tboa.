import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from datetime import datetime

# --- Setup Streamlit page
st.set_page_config(page_title="Phân tích sản phẩm bán chậm", layout="wide")

# --- List of 10 slow-selling products
slow_selling_products = [
    "Cà phê sữa The Coffee House 180ml (1 lon)",
    "Cà phê sữa đá hòa tan The Coffee House bịch 25 gói x 22g",
    "Cà phê sữa đá miền Nam Cà Phê Phố túi 30 gói x 24g",
    "Cà phê sữa nhà làm Cà Phê Phố túi 30 gói x 28g",
    "Cà phê rang xay culi Highlands Coffee gói 200g",
    "Cà phê hoà tan nguyên chất 114 UCC hộp 20g (10 gói x 2g) (1 Hộp)",
    "Cà phê hoà tan nguyên chất Sumiyaki UCC hộp 20g (10 gói x 2g) (1 Hộp)",
    "Cà phê hoà tan nguyên chất 117 UCC hộp 20g (10 gói x 2g) (1 Hộp)",
    "Cà phê hoà tan black 2IN1 K-Coffee hộp 15 gói x 17g (1 Hộp)",
    "Cà phê hoà tan pha lạnh 3in1 vị nguyên bản UCC hộp 250g (10 gói x 25g) (1 Hộp)"
]

# --- Load Excel file
uploaded_file = st.file_uploader("📂 Tải lên file Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Trang tính1")
    df = df[df["name"].isin(slow_selling_products)].copy()

    # --- Helper functions
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return None

    def sum_stock_decreased_in_range(stock_history_str, start_date, end_date):
        try:
            history = json.loads(stock_history_str).get("stock_history", [])
            start = parse_date(start_date)
            end = parse_date(end_date)
            total = 0
            for entry in history:
                entry_date = parse_date(entry.get("date"))
                if entry_date and start <= entry_date <= end:
                    total += float(entry.get("stock_decreased", 0))
            return total
        except:
            return 0

    # --- Tính số lượng theo tuần
    df["Tuần 1"] = df["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-07", "2025-03-14"))
    df["Tuần 2"] = df["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-15", "2025-03-22"))
    df["Tuần 3"] = df["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-23", "2025-03-28"))
    df["Cả tháng"] = df["Tuần 1"] + df["Tuần 2"] + df["Tuần 3"]

    # --- Dropdown lựa chọn kỳ
    period = st.selectbox("📊 Chọn kỳ phân tích", ["Tuần 1", "Tuần 2", "Tuần 3", "Cả tháng"])

    # --- Lọc dữ liệu
    threshold = 25
    if period in ["Tuần 1", "Tuần 2", "Tuần 3"]:
        data_filtered = df[(df[period] > 0) & (df[period] < threshold)][["name", period]].sort_values(by=period, ascending=False)
    else:
        mask = (
            ((df["Tuần 1"] > 0) & (df["Tuần 1"] < threshold)) |
            ((df["Tuần 2"] > 0) & (df["Tuần 2"] < threshold)) |
            ((df["Tuần 3"] > 0) & (df["Tuần 3"] < threshold))
        )
        df_filtered = df[mask].copy()
        df_filtered["Cả tháng"] = df_filtered["Tuần 1"] + df_filtered["Tuần 2"] + df_filtered["Tuần 3"]
        data_filtered = df_filtered[["name", "Cả tháng"]].sort_values(by="Cả tháng", ascending=False)

    # --- Hiển thị kết quả
    if data_filtered.empty:
        st.success(f"✅ Không có sản phẩm nào bán chậm trong {period}.")
    else:
        st.subheader(f"📌 Biểu đồ sản phẩm bán chậm - {period}")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=data_filtered, x=period, y="name", palette="Set2", ax=ax)

        for i, v in enumerate(data_filtered[period]):
            ax.text(v + 0.2, i, str(int(v)), va='center', color='black', fontweight='bold')

        ax.set_xlabel("Số lượng bán")
        ax.set_ylabel("Tên sản phẩm")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        st.pyplot(fig)

        st.info(f"🧾 Tổng số lượng bán trong kỳ: **{int(data_filtered[period].sum())}** sản phẩm.")

