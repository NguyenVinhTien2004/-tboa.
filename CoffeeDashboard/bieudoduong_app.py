import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình trang Streamlit
st.set_page_config(layout="wide", page_title="Dashboard Doanh Thu", page_icon="📈")
st.title("📊 Biểu Đồ Doanh Thu Top 5 Sản Phẩm")

# Đọc dữ liệu từ file
@st.cache_data  # Cache để tăng hiệu suất
def load_data():
    try:
        df = pd.read_csv("kf_coffee.csv")
        return df
    except Exception as e:
        st.error(f"Không thể đọc file dữ liệu: {e}")
        return None

df = load_data()

if df is not None:
    # Hàm tính tổng số lượng bán từ stock_history
    def total_sold_from_history(stock_history_str):
        try:
            history = json.loads(stock_history_str).get("stock_history", [])
            return sum(float(entry.get("stock_decreased", 0)) for entry in history)
        except:
            return 0

    # Tính toán dữ liệu
    df['so_luong_ban'] = df['stock_history'].apply(total_sold_from_history)
    df['doanh_thu'] = df['so_luong_ban'] * df['price']
    revenue_by_product = df.groupby('name')['doanh_thu'].sum().sort_values(ascending=False).head(5)

    # Tạo container cho biểu đồ
    with st.container():
        # Tạo figure cho biểu đồ
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#F5F5F5")
        sns.set_style("whitegrid")

        # Vẽ biểu đồ đường
        sns.lineplot(
            x=revenue_by_product.index,
            y=revenue_by_product.values,
            marker='o',
            markersize=10,
            linewidth=3,
            color='#2196F3',
            markeredgecolor='black',
            markeredgewidth=1,
            ax=ax
        )

        # Hiệu ứng shadow
        sns.lineplot(
            x=revenue_by_product.index,
            y=revenue_by_product.values,
            marker='o',
            markersize=8,
            linewidth=5,
            color='#BBDEFB',
            alpha=0.5,
            ax=ax
        )

        # Thêm nhãn giá trị
        y_max = max(revenue_by_product.values)
        for i, (x, y) in enumerate(zip(revenue_by_product.index, revenue_by_product.values)):
            ax.text(
                x,
                y + (y_max * 0.01),
                f"{int(y):,} VNĐ",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='#4CAF50'
            )

        # Điều chỉnh giới hạn trục y
        ax.set_ylim(0, y_max * 1.3)

        # Tuỳ chỉnh giao diện
        ax.set_title("Top 5 Sản Phẩm Doanh Thu Cao Nhất", fontsize=16, pad=20)
        ax.set_xlabel("Sản Phẩm", fontsize=12)
        ax.set_ylabel("Doanh Thu (VNĐ)", fontsize=12)
        ax.set_xticklabels(revenue_by_product.index, rotation=45, ha='right', fontsize=10)
        ax.set_facecolor("#FFFFFF")
        
        # Định dạng khung biểu đồ
        for spine in ax.spines.values():
            spine.set_color('#B0BEC5')
            spine.set_linewidth(1.2)

        ax.yaxis.grid(True, linestyle='--', color='#E0E0E0', alpha=0.7)
        ax.xaxis.grid(False)

        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)

    # Hiển thị bảng dữ liệu
    with st.expander("📊 Xem dữ liệu chi tiết"):
        st.dataframe(
            revenue_by_product.reset_index().rename(columns={'name': 'Sản phẩm', 'doanh_thu': 'Doanh thu (VNĐ)'}),
            column_config={
                "Doanh thu (VNĐ)": st.column_config.NumberColumn(format="%,d")
            },
            use_container_width=True
        )

    # Thêm thông tin phụ
    st.caption("💡 Dữ liệu được cập nhật lần cuối: " + pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"))
else:
    st.warning("Vui lòng tải lên file dữ liệu hợp lệ để hiển thị biểu đồ")