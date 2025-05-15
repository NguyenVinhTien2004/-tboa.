import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Cấu hình trang tổng thể
st.set_page_config(
    layout="wide",
    page_title="Dashboard Phân Tích Cà Phê",
    page_icon="☕"
)

# Tiêu đề tổng
st.title("☕ Dashboard Phân Tích Kinh Doanh Cà Phê")
st.markdown("---")

# Tạo các tab
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 KPI Tổng Quan", 
    "📈 Top 10 Bán Chạy", 
    "📉 Sản Phẩm Bán Chậm", 
    "📉 Biểu Đồ Đường", 
    "🟣 Biểu Đồ Tròn"
])

# ========================================
# TAB 1: KPI Tổng Quan (từ KPI_app.py)
# ========================================
with tab1:
    st.header("📊 KPI Dashboard Doanh Số Cà Phê")

    # Đọc dữ liệu
    try:
        df_kpi = pd.read_excel("kf_coffee (1).xlsx", sheet_name="Trang tính1")
    except:
        st.error("Không tìm thấy file dữ liệu kf_coffee (1).xlsx")
        st.stop()

    # Hàm parse date
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return None

    # Hàm tính số lượng bán
    def sum_stock_decreased_in_range(stock_history_str, start_date, end_date):
        try:
            history = json.loads(stock_history_str).get("stock_history", [])
            start = parse_date(start_date)
            end = parse_date(end_date)
            if not start or not end:
                return 0
            total = 0
            for entry in history:
                entry_date = parse_date(entry.get("date"))
                if entry_date and start <= entry_date <= end:
                    total += float(entry.get("stock_decreased", 0))
            return total
        except:
            return 0

    # Tính toán các chỉ số
    df_kpi["Tuan_1"] = df_kpi["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-07", "2025-03-14"))
    df_kpi["Tuan_2"] = df_kpi["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-15", "2025-03-22"))
    df_kpi["Tuan_3"] = df_kpi["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-23", "2025-03-28"))
    df_kpi["Ca_thang"] = df_kpi["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-07", "2025-03-28"))

    # Tính doanh thu theo từng kỳ
    period_order = ["Tuan_1", "Tuan_2", "Tuan_3", "Ca_thang"]
    for period in period_order:
        df_kpi[f"doanh_thu_{period}"] = df_kpi[period] * df_kpi["price"]

    period_label = {
        "Tuan_1": "Tuần 1 (07/03 → 14/03)",
        "Tuan_2": "Tuần 2 (15/03 → 22/03)",
        "Tuan_3": "Tuần 3 (23/03 → 28/03)",
        "Ca_thang": "Cả tháng (07/03 → 28/03)"
    }

    selected_period = st.selectbox("Chọn kỳ:", options=period_order,
                                   format_func=lambda x: period_label[x])

    # Gán doanh thu của kỳ đang chọn vào cột 'doanh_thu'
    df_kpi["doanh_thu"] = df_kpi[f"doanh_thu_{selected_period}"]
    total_units_sold = df_kpi[selected_period].sum()
    
    # Tính tổng doanh thu dựa trên số liệu cung cấp
    revenue_dict = {
        "Tuan_1": 70595200,  # Tuần 1
        "Tuan_2": 68469400,  # Tuần 2
        "Tuan_3": 17212500,  # Tuần 3
        "Ca_thang": 127622000  # Cả tháng
    }
    total_revenue = revenue_dict[selected_period]

    top_product = df_kpi.sort_values(by="doanh_thu", ascending=False).iloc[0]["name"]

    # Tính tăng trưởng so với kỳ trước
    current_idx = period_order.index(selected_period)
    if current_idx > 0:
        prev_period = period_order[current_idx - 1]
        prev_units_sold = df_kpi[prev_period].sum()
        growth = ((total_units_sold - prev_units_sold) / prev_units_sold * 100) if prev_units_sold > 0 else 0
        growth_text = f"{growth:+.1f}%"
    else:
        growth_text = "N/A"

    growth_title = "Tăng trưởng so với tuần trước" if selected_period != "Ca_thang" else "Tăng trưởng của cả tháng"

    # Hiển thị thẻ KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Tổng doanh thu", f"{int(total_revenue):,} VNĐ")
    col2.metric("📦 Sản phẩm bán ra", f"{int(total_units_sold):,}")

    with col3:
        st.markdown("🔥 **Sản phẩm có doanh thu cao nhất**", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:18px; white-space:normal'>{top_product}</div>", unsafe_allow_html=True)

    col4.metric("📊 " + growth_title, growth_text)

    # Biểu đồ top sản phẩm
    st.subheader(f"Top sản phẩm theo doanh thu - {period_label[selected_period]}")
    top_products = df_kpi.sort_values(by="doanh_thu", ascending=False).head(10)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(top_products["name"], top_products["doanh_thu"], color="#4CAF50")
    ax1.set_ylabel("Doanh thu (VNĐ)")
    ax1.set_xticklabels(top_products["name"], rotation=45, ha='right')

    # Thêm số liệu lên từng cột
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.01 * height,
                 f"{int(height):,}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    st.pyplot(fig1)
# ========================================
# TAB 2: Top 10 Bán Chạy (từ top10_app.py)
# ========================================
with tab2:
    st.header("🏆 Top 10 Sản Phẩm Bán Chạy Nhất")
    
    try:
        df_top10 = pd.read_csv("kf_coffee.csv")
    except:
        st.error("Không tìm thấy file dữ liệu kf_coffee.csv")
        st.stop()
    
    # Hàm tính tổng số lượng bán
    def total_stock_decreased(history_str):
        try:
            history = json.loads(history_str).get("stock_history", [])
            return sum(float(entry.get("stock_decreased", 0)) for entry in history)
        except:
            return 0
    
    df_top10["stock_decreased"] = df_top10["stock_history"].apply(total_stock_decreased)
    top_products = df_top10.groupby('name')['stock_decreased'].sum().sort_values(ascending=False).head(10)
    
    # Vẽ biểu đồ
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax2)
    
    for i, v in enumerate(top_products.values):
        ax2.text(v + 5, i, str(int(v)), color='black', va='center', fontweight='bold')
    
    ax2.set_xlabel("Số lượng bán")
    ax2.set_ylabel("Sản phẩm")
    st.pyplot(fig2)

# ========================================
# TAB 3: Sản Phẩm Bán Chậm (từ bancham_app.py)
# ========================================
with tab3:
    st.header("⚠️ Phân Tích Sản Phẩm Bán Chậm")
    
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
    
    try:
        df_slow = pd.read_excel("kf_coffee (1).xlsx", sheet_name="Trang tính1")
        df_slow = df_slow[df_slow["name"].isin(slow_selling_products)].copy()
    except:
        st.error("Không tìm thấy file dữ liệu")
        st.stop()
    
    # Hàm hỗ trợ
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
    
    # Tính toán
    df_slow["Tuần 1"] = df_slow["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-07", "2025-03-14"))
    df_slow["Tuần 2"] = df_slow["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-15", "2025-03-22"))
    df_slow["Tuần 3"] = df_slow["stock_history"].apply(lambda x: sum_stock_decreased_in_range(x, "2025-03-23", "2025-03-28"))
    df_slow["Cả tháng"] = df_slow["Tuần 1"] + df_slow["Tuần 2"] + df_slow["Tuần 3"]
    
    period = st.selectbox("Chọn kỳ phân tích:", ["Tuần 1", "Tuần 2", "Tuần 3", "Cả tháng"])
    
    # Lọc dữ liệu
    threshold = 25
    if period in ["Tuần 1", "Tuần 2", "Tuần 3"]:
        data_filtered = df_slow[(df_slow[period] > 0) & (df_slow[period] < threshold)][["name", period]].sort_values(by=period, ascending=False)
    else:
        mask = (
            ((df_slow["Tuần 1"] > 0) & (df_slow["Tuần 1"] < threshold)) |
            ((df_slow["Tuần 2"] > 0) & (df_slow["Tuần 2"] < threshold)) |
            ((df_slow["Tuần 3"] > 0) & (df_slow["Tuần 3"] < threshold)))
        df_filtered = df_slow[mask].copy()
        data_filtered = df_filtered[["name", "Cả tháng"]].sort_values(by="Cả tháng", ascending=False)
    
    # Tính tổng số lượng bán ra cho kỳ được chọn
    total_period = data_filtered[period].sum() if not data_filtered.empty else 0
    
    # Hiển thị kết quả
    if data_filtered.empty:
        st.success(f"✅ Không có sản phẩm nào bán chậm trong {period}.")
    else:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=data_filtered, x=period, y="name", palette="Set2", ax=ax3)
        
        for i, v in enumerate(data_filtered[period]):
            ax3.text(v + 0.2, i, str(int(v)), va='center', color='black', fontweight='bold')
        
        ax3.set_xlabel("Số lượng bán")
        ax3.set_ylabel("Tên sản phẩm")
        st.pyplot(fig3)
        
        # Hiển thị tổng số lượng bán ra cho kỳ được chọn
        st.info(
            f"📌 Tổng sản phẩm bán ra (trong nhóm bán chậm): **{int(total_period):,}** đơn vị."
        )

# ========================================
# TAB 4: Biểu Đồ Đường (từ bieudoduong_app.py)
# ========================================
with tab4:
    st.header("📈 Biểu Đồ Doanh Thu Top 5 Sản Phẩm")

    try:
        df_line = pd.read_csv("kf_coffee.csv")
    except:
        st.error("Không tìm thấy file dữ liệu kf_coffee.csv")
        st.stop()

    # Hàm tính tổng số lượng bán
    def total_sold_from_history(stock_history_str):
        try:
            history = json.loads(stock_history_str).get("stock_history", [])
            return sum(float(entry.get("stock_decreased", 0)) for entry in history)
        except:
            return 0

    # Tính toán doanh thu
    df_line['so_luong_ban'] = df_line['stock_history'].apply(total_sold_from_history)
    df_line['doanh_thu'] = df_line['so_luong_ban'] * df_line['price']
    revenue_by_product = df_line.groupby('name')['doanh_thu'].sum().sort_values(ascending=False).head(5)

    # Vẽ biểu đồ
    fig4, ax4 = plt.subplots(figsize=(20, 10), facecolor="#F5F5F5")
    sns.set_style("whitegrid")

    # Vẽ đường chính
    sns.lineplot(x=revenue_by_product.index, y=revenue_by_product.values, marker='o',
                 markersize=15, linestyle='-', linewidth=3, color='#2196F3',
                 markeredgecolor='black', markeredgewidth=1, ax=ax4)

    # Vẽ bóng nền
    sns.lineplot(x=revenue_by_product.index, y=revenue_by_product.values, marker='o',
                 markersize=12, linestyle='-', linewidth=5, color='#BBDEFB', alpha=0.5, ax=ax4)

    # Thêm nhãn số lên điểm
    for i, (x, y) in enumerate(zip(revenue_by_product.index, revenue_by_product.values)):
        ax4.text(i, y + y*0.01, f"{int(y):,} VNĐ", ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='#4CAF50')

    # Tùy chỉnh biểu đồ
    ax4.set_title("Doanh Thu Top 5 Sản Phẩm (Line Chart)", fontsize=18, fontweight='bold', pad=20, color='#333333')
    ax4.set_xlabel("Sản Phẩm", fontsize=14, fontweight='bold', color='#333333')
    ax4.set_ylabel("Doanh Thu (VNĐ)", fontsize=14, fontweight='bold', color='#333333')
    ax4.set_xticklabels(revenue_by_product.index, rotation=45, fontsize=12, fontweight='bold', color='#333333')
    ax4.tick_params(axis='y', labelsize=12)

    # Viền khung biểu đồ
    for spine in ax4.spines.values():
        spine.set_color('#B0BEC5')
        spine.set_linewidth(1.5)

    # Đường lưới
    ax4.yaxis.grid(True, linestyle='--', color='#E0E0E0', alpha=0.7)
    ax4.xaxis.grid(False)

    st.pyplot(fig4)

    # ========================================
# TAB 5: Biểu Đồ Tròn
# ========================================
with tab5:
    st.header("🟣 Biểu Đồ Tròn Phân Phối Sản Phẩm")

    import plotly.graph_objects as go

    # Dữ liệu loại bao bì
    packaging_types = ['Lon', 'Hộp', 'Túi', 'Gói', 'Bịch', 'Ly']
    packaging_counts = [3, 23, 6, 13, 4, 1]
    packaging_colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1', '#F4E04D']

    # Dữ liệu loại cà phê
    coffee_types = ['Hòa tan', 'Rang xay', 'Sữa']
    coffee_counts = [21, 9, 26]
    coffee_colors = ['#FF6F61', '#6B5B95', '#88B04B']

    # Chọn loại biểu đồ
    selected_option = st.selectbox(
        'Chọn biểu đồ:',
        ['Phân phối sản phẩm theo loại bao bì', 'Phân phối sản phẩm theo loại cà phê']
    )

    # Gán dữ liệu tương ứng
    if selected_option == 'Phân phối sản phẩm theo loại bao bì':
        labels = packaging_types
        values = packaging_counts
        colors = packaging_colors
        title = 'Phân phối sản phẩm theo loại bao bì'
    else:
        labels = coffee_types
        values = coffee_counts
        colors = coffee_colors
        title = 'Phân phối sản phẩm theo loại cà phê'

    # Vẽ biểu đồ
        # Vẽ biểu đồ
    fig5 = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            textinfo='label+percent',
            hoverinfo='label+value',
            marker=dict(colors=colors),
            insidetextorientation='radial',
            textfont=dict(size=18)  # 👉 Cỡ chữ bên trong biểu đồ
        )]
    )

    fig5.update_layout(
        title=title,
        showlegend=True,
        legend=dict(font=dict(size=22))# 👉 Cỡ chữ phần chú thích
    )

    st.plotly_chart(fig5, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Công ty TNHH LIBERAIN")