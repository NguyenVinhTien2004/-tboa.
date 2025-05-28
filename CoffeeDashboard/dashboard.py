import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import altair as alt
from datetime import datetime, date
from pymongo import MongoClient
from bson import ObjectId

# ----- Hàm kết nối MongoDB với xử lý lỗi -----
@st.cache_resource
def init_connection():
    try:
        # Sử dụng giá trị mặc định nếu không có secrets.toml
        mongo_uri = "mongodb+srv://nkq5986:1234567Aa@cluster0.lwe3e.mongodb.net/product_db"
        db_name = "db_kf"
        collection_name = "kf_new"
        
        # Thử đọc từ secrets.toml nếu có
        try:
            if hasattr(st, 'secrets') and 'mongo' in st.secrets:
                mongo_uri = st.secrets["mongo"].get("host", mongo_uri)
                db_name = st.secrets["mongo"].get("database", db_name)
                collection_name = st.secrets["mongo"].get("collection", collection_name)
        except:
            pass
            
        client = MongoClient(mongo_uri)
        return client, db_name, collection_name
    except Exception as e:
        st.error(f"Không thể kết nối MongoDB: {str(e)}")
        return None, None, None

# ----- Load Data từ MongoDB với xử lý lỗi -----
@st.cache_data(ttl=3600)
def load_data_from_mongodb():
    client, db_name, collection_name = init_connection()
    if client is None:
        return pd.DataFrame(), date(2025, 3, 5), date(2025, 5, 25)
    
    try:
        db = client[db_name]
        collection = db[collection_name]
        data = list(collection.find({}))
        
        if not data:
            st.warning("Không tìm thấy dữ liệu trong MongoDB")
            return pd.DataFrame(), date(2025, 3, 5), date(2025, 5, 25)

        all_data = []
        all_dates = []
        
        for product in data:
            try:
                # Kiểm tra các trường bắt buộc
                if 'price' not in product or 'stock_history' not in product:
                    continue
                    
                # Xử lý lịch sử
                stock_history = product.get('stock_history', [])
                if not isinstance(stock_history, list):
                    continue
                
                # Kiểm tra dữ liệu bán hàng hợp lệ
                stock_decreased_values = []
                for entry in stock_history:
                    if isinstance(entry, dict):
                        try:
                            val = float(entry.get('stock_decreased', 0))
                            stock_decreased_values.append(val)
                        except:
                            continue
                
                has_meaningful_sales = any(abs(val) > 1e-10 for val in stock_decreased_values)
                has_valid_data = all(abs(val - round(val)) <= 1e-10 
                                for val in stock_decreased_values if abs(val) > 1e-10)
                
                if not has_meaningful_sales or not has_valid_data:
                    continue
                
                # Tính toán các chỉ số
                total_sold = sum(float(entry.get('stock_decreased', 0)) 
                             for entry in stock_history if isinstance(entry, dict))
                
                try:
                    current_price = float(product['price'])
                except:
                    current_price = 0
                
                # Thu thập ngày tháng
                for entry in stock_history:
                    if isinstance(entry, dict) and 'date' in entry and entry['date']:
                        try:
                            entry_date = datetime.strptime(entry['date'], '%Y-%m-%d').date()
                            all_dates.append(entry_date)
                        except:
                            continue
                
                # Thêm vào danh sách
                all_data.append({
                    'id': str(product.get('_id', '')),
                    'name': product.get('name', ''),
                    'category': product.get('category', ''),
                    'price': current_price,
                    'promotion': product.get('promotion', ''),
                    'total_sold': total_sold,
                    'revenue': current_price * total_sold,
                    'sales_history': product.get('sales_history', []),
                    'stock_history': stock_history,
                    'price_history': product.get('price_history', []),
                    'source_file': 'MongoDB'
                })
                
            except Exception as e:
                st.error(f"Lỗi xử lý sản phẩm {product.get('_id', '')}: {str(e)}")
                continue
        
        # Tạo DataFrame
        df = pd.DataFrame(all_data) if all_data else pd.DataFrame()
        
        # Xác định phạm vi ngày
        if not all_dates:
            st.warning("Không tìm thấy ngày hợp lệ. Sử dụng ngày mặc định.")
            min_date = date(2025, 3, 5)
            max_date = date(2025, 5, 25)
        else:
            min_date = min(all_dates)
            max_date = max(all_dates)
            
        return df, min_date, max_date
        
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu từ MongoDB: {str(e)}")
        return pd.DataFrame(), date(2025, 3, 5), date(2025, 5, 25)
    finally:
        if client:
            client.close()

# ----- Hàm phân cụm với xử lý lỗi -----
def apply_clustering(df):
    if len(df) > 0:
        try:
            # Đảm bảo cột price là số
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.dropna(subset=['price'])
            
            prices = df[['price']]
            if len(prices) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=42)
                labels = kmeans.fit_predict(prices)
                cluster_centers = kmeans.cluster_centers_.flatten()
                label_order = np.argsort(cluster_centers)
                label_mapping = {old: new for new, old in enumerate(label_order)}
                df.loc[prices.index, 'segment'] = [label_mapping[label] for label in labels]
                segment_label_map = {0: 'Thấp', 1: 'Trung', 2: 'Cao'}
                df['segment'] = df['segment'].map(segment_label_map)
            else:
                st.warning("Không đủ dữ liệu để phân cụm. Gán mặc định 'Thấp'")
                df['segment'] = 'Thấp'
        except Exception as e:
            st.error(f"Lỗi khi phân cụm: {str(e)}")
            df['segment'] = 'Không xác định'
    else:
        df['segment'] = None
    return df

# ----- Hàm lọc theo ngày với xử lý lỗi -----
def filter_by_date_range(df, start_date, end_date):
    filtered_data = []
    for _, row in df.iterrows():
        try:
            if not isinstance(row.get('stock_history', []), list):
                continue
                
            total_sold = 0
            total_stock_increased = 0
            for entry in row['stock_history']:
                try:
                    if not isinstance(entry, dict):
                        continue
                        
                    entry_date = datetime.strptime(entry.get('date', ''), '%Y-%m-%d').date()
                    if start_date <= entry_date <= end_date:
                        total_sold += float(entry.get('stock_decreased', 0))
                        total_stock_increased += float(entry.get('stock_increased', 0))
                except:
                    continue
            
            filtered_data.append({
                'id': row['id'],
                'name': row['name'],
                'category': row['category'],
                'price': row['price'],
                'promotion': row['promotion'],
                'quantity_sold': total_sold,
                'stock_remaining': total_stock_increased,
                'revenue': total_sold * row['price'],
                'stock_revenue': total_stock_increased * row['price'],
                'segment': row['segment'],
                'stock_history': row['stock_history'],
                'source_file': row['source_file']
            })
        except Exception as e:
            st.error(f"Lỗi khi lọc hàng {_}: {str(e)}")
            continue
    return pd.DataFrame(filtered_data)

# ----- Giao diện chính -----
st.set_page_config(layout="wide", page_title="Dashboard Phân Khúc Doanh Thu và Tồn Kho - KingFoodMart")

# Load data
df, min_date, max_date = load_data_from_mongodb()

if df.empty:
    st.error("""
    Không có dữ liệu từ MongoDB. Vui lòng kiểm tra:
    1. Kết nối MongoDB
    2. Tên database và collection
    3. Dữ liệu trong collection có đúng định dạng không?
    """)
    st.stop()

# Kiểm tra các cột bắt buộc
required_columns = ['price', 'stock_history', 'category']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"Thiếu các cột bắt buộc: {', '.join(missing_cols)}")
    st.stop()

# Xử lý dữ liệu
df = df.dropna(subset=['price'])
df = apply_clustering(df)

# ----- Sidebar Filters -----
st.sidebar.header("Bộ lọc")
selected_category = st.sidebar.selectbox("Chọn danh mục", ['Tất cả'] + sorted(df['category'].unique()))
filtered_df = df if selected_category == 'Tất cả' else df[df['category'] == selected_category]

segment_options = ['Tất cả'] + sorted(filtered_df['segment'].dropna().unique())
selected_segment = st.sidebar.selectbox("Phân khúc", segment_options)
if selected_segment != 'Tất cả':
    filtered_df = filtered_df[filtered_df['segment'] == selected_segment]

product_options = ['Tất cả'] + sorted(filtered_df['name'].unique())
selected_product = st.sidebar.selectbox("Sản phẩm", product_options)
if selected_product != 'Tất cả':
    filtered_df = filtered_df[filtered_df['name'] == selected_product]

display_mode = st.sidebar.selectbox("Chế độ hiển thị", ["Bán hàng", "Tồn kho"])

default_start = max(min_date, date(2025, 3, 5))
default_end = min(max_date, date(2025, 5, 18))

start_date = st.sidebar.date_input("Ngày bắt đầu", value=default_start, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("Ngày kết thúc", value=default_end, min_value=min_date, max_value=max_date)

filtered_df = filter_by_date_range(filtered_df, start_date, end_date)

# ----- Main Dashboard -----
st.title("📊 Dashboard Phân Khúc Doanh Thu và Tồn Kho - KingFoodMart")

st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] > div { gap: 20px !important; }
div[data-testid="stMetric"] { min-width: 200px !important; padding: 15px !important; background-color: #f8f9fa; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); flex: 1; }
[data-testid="stMetricValue"] > div { font-size: 22px !important; font-weight: bold; color: #2c3e50; white-space: normal !important; overflow: visible !important; text-overflow: unset !important; line-height: 1.3; height: auto !important; }
[data-testid="stMetricLabel"] > div { font-size: 16px !important; font-weight: 600; color: #7f8c8d; margin-bottom: 10px !important; white-space: normal !important; overflow: visible !important; text-overflow: unset !important; height: auto !important; }
.stDataFrame { font-size: 14px; }
.vega-embed { padding-bottom: 20px !important; }
@media (max-width: 768px) { div[data-testid="stHorizontalBlock"] { flex-direction: column !important; } div[data-testid="stMetric"] { min-width: 100% !important; margin-bottom: 15px !important; } }
</style>
""", unsafe_allow_html=True)

# ----- KPI Metrics -----
st.subheader("📈Tổng Quan")
col1, col2, col3, col4 = st.columns(4)

def format_number(num):
    return f"{num:,.0f}".replace(",", ".")

def shorten_product_name(name, max_length=20):
    if len(str(name)) > max_length:
        return str(name)[:max_length-3] + "..."
    return str(name)

total_revenue = filtered_df['revenue'].sum() if not filtered_df.empty else 0
total_stock_revenue = filtered_df['stock_revenue'].sum() if not filtered_df.empty else 0
total_quantity = filtered_df['quantity_sold'].sum() if not filtered_df.empty else 0
total_stock = filtered_df['stock_remaining'].sum() if not filtered_df.empty else 0
avg_price = filtered_df['price'].mean() if not filtered_df.empty else 0

with col1:
    if display_mode == "Bán hàng":
        st.metric("Tổng Doanh Thu", f"{format_number(total_revenue)} VND")
    else:
        st.metric("Tổng Doanh Thu Tồn Kho", f"{format_number(total_stock_revenue)} VND")

with col2:
    if display_mode == "Bán hàng":
        st.metric("Tổng Số Lượng Bán", format_number(total_quantity))
    else:
        st.metric("Tổng Tồn Kho", format_number(total_stock))

with col3:
    st.metric("Giá Trung Bình", f"{format_number(avg_price)} VND")

with col4:
    if display_mode == "Bán hàng":
        top_product = (
            filtered_df.loc[filtered_df['quantity_sold'].idxmax()]['name']
            if not filtered_df.empty and 'quantity_sold' in filtered_df.columns and filtered_df['quantity_sold'].sum() > 0 else "N/A"
        )
        st.metric("Sản Phẩm Bán Chạy", shorten_product_name(top_product))
    else:
        top_stock_product = (
            filtered_df.loc[filtered_df['stock_remaining'].idxmax()]['name']
            if not filtered_df.empty and 'stock_remaining' in filtered_df.columns and filtered_df['stock_remaining'].sum() > 0 else "N/A"
        )
        st.metric("Sản Phẩm Tồn Kho Nhiều Nhất", shorten_product_name(top_stock_product))

# ----- Top Sản Phẩm -----
st.subheader("🏆 Top Sản Phẩm")

if not filtered_df.empty:
    if display_mode == "Bán hàng":
        top_products = filtered_df.sort_values('quantity_sold', ascending=False).head(5)
        slow_products = filtered_df[filtered_df['quantity_sold'] > 0].sort_values('quantity_sold').head(5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 5 Sản Phẩm Bán Chạy**")
            if not top_products.empty:
                chart_top = alt.Chart(top_products).mark_bar().encode(
                    x=alt.X('quantity_sold:Q', title='Số lượng bán'),
                    y=alt.Y('name:N', title='Tên sản phẩm', sort='-x'),
                    color=alt.Color('quantity_sold:Q', legend=None, scale=alt.Scale(scheme='greens')),
                    tooltip=['name:N', 'quantity_sold:Q', 'revenue:Q']
                ).properties(height=300)
                st.altair_chart(chart_top, use_container_width=True)
            else:
                st.info("Không có sản phẩm nào bán chạy")
        
        with col2:
            st.markdown("**Top 5 Sản Phẩm Bán Chậm**")
            if not slow_products.empty:
                chart_slow = alt.Chart(slow_products).mark_bar().encode(
                    x=alt.X('quantity_sold:Q', title='Số lượng bán'),
                    y=alt.Y('name:N', title='Tên sản phẩm', sort='x'),
                    color=alt.Color('quantity_sold:Q', legend=None, scale=alt.Scale(scheme='reds')),
                    tooltip=['name:N', 'quantity_sold:Q', 'revenue:Q']
                ).properties(height=300)
                st.altair_chart(chart_slow, use_container_width=True)
            else:
                st.info("Không có sản phẩm nào bán chậm")
    else:
        top_stock_products = filtered_df.sort_values('stock_remaining', ascending=False).head(5)
        low_stock_products = filtered_df[filtered_df['stock_remaining'] > 0].sort_values('stock_remaining').head(5)
        
        st.markdown("**Top 5 Sản Phẩm Nhập Kho Nhiều Nhất**")
        if not top_stock_products.empty:
            chart_top_stock = alt.Chart(top_stock_products).mark_bar().encode(
                x=alt.X('stock_remaining:Q', title='Số lượng tồn kho'),
                y=alt.Y('name:N', title='Tên sản phẩm', sort='-x'),
                color=alt.Color('stock_remaining:Q', legend=None, scale=alt.Scale(scheme='greens')),
                tooltip=['name:N', 'stock_remaining:Q', 'stock_revenue:Q']
            ).properties(height=300)
            
            text_top = chart_top_stock.mark_text(
                align='left',
                baseline='middle',
                dx=3
            ).encode(
                text=alt.Text('stock_remaining:Q', format='.0f')
            )
            
            st.altair_chart(chart_top_stock + text_top, use_container_width=True)
        else:
            st.info("Không có sản phẩm nào có tồn kho")
        
        st.markdown("**Top 5 Sản Phẩm Nhập Kho Ít Nhất**")
        if not low_stock_products.empty:
            chart_low_stock = alt.Chart(low_stock_products).mark_bar().encode(
                x=alt.X('stock_remaining:Q', title='Số lượng tồn kho'),
                y=alt.Y('name:N', title='Tên sản phẩm', sort='x'),
                color=alt.Color('stock_remaining:Q', legend=None, scale=alt.Scale(scheme='reds')),
                tooltip=['name:N', 'stock_remaining:Q', 'stock_revenue:Q']
            ).properties(height=300)
            
            text_low = chart_low_stock.mark_text(
                align='left',
                baseline='middle',
                dx=3
            ).encode(
                text=alt.Text('stock_remaining:Q', format='.0f')
            )
            
            st.altair_chart(chart_low_stock + text_low, use_container_width=True)
        else:
            st.info("Không có sản phẩm nào có tồn kho thấp")
else:
    st.warning("Không có dữ liệu để hiển thị")

# ----- Biểu Đồ Theo Ngày -----
if display_mode == "Bán hàng":
    st.subheader("📈 Biểu Đồ Doanh Thu Theo Ngày")
else:
    st.subheader("📈 Biểu Đồ Tồn Kho Theo Ngày")

if not filtered_df.empty:
    daily_data = []
    for _, row in filtered_df.iterrows():
        for entry in row['stock_history']:
            try:
                entry_date = datetime.strptime(entry.get('date', ''), '%Y-%m-%d').date()
                if start_date <= entry_date <= end_date:
                    daily_data.append({
                        'date': entry_date,
                        'quantity_sold': float(entry.get('stock_decreased', 0)),
                        'stock_remaining': float(entry.get('stock_increased', 0)),
                        'name': row['name'],
                        'price': row['price']
                    })
            except:
                continue

    if daily_data:
        daily_df = pd.DataFrame(daily_data)
        if selected_product != 'Tất cả':
            daily_df = daily_df[daily_df['name'] == selected_product]

        if not daily_df.empty:
            if display_mode == "Bán hàng":
                daily_agg = daily_df.groupby('date').agg({
                    'quantity_sold': 'sum',
                    'price': 'mean'
                }).reset_index()
                daily_agg['revenue'] = (daily_agg['quantity_sold'] * daily_agg['price']).astype(int)

                revenue_chart = alt.Chart(daily_agg).mark_line(point=True).encode(
                    x=alt.X('date:T', title='Ngày'),
                    y=alt.Y('revenue:Q', title='Doanh Thu (VND)'),
                    tooltip=['date:T', 'revenue:Q']
                ).properties(height=300)
                
                quantity_chart = alt.Chart(daily_agg).mark_bar().encode(
                    x=alt.X('date:T', title='Ngày'),
                    y=alt.Y('quantity_sold:Q', title='Số Lượng Bán'),
                    tooltip=['date:T', 'quantity_sold:Q']
                ).properties(height=300)
                
                st.altair_chart(revenue_chart, use_container_width=True)
                st.altair_chart(quantity_chart, use_container_width=True)
            else:
                daily_stock_agg = daily_df.groupby('date').agg({
                    'stock_remaining': 'sum',
                    'price': 'mean'
                }).reset_index()
                daily_stock_agg['stock_revenue'] = (daily_stock_agg['stock_remaining'] * daily_stock_agg['price']).astype(int)

                stock_revenue_chart = alt.Chart(daily_stock_agg).mark_line(point=True).encode(
                    x=alt.X('date:T', title='Ngày'),
                    y=alt.Y('stock_revenue:Q', title='Doanh Thu Tồn Kho (VND)'),
                    tooltip=['date:T', 'stock_revenue:Q']
                ).properties(height=300)
                
                stock_quantity_chart = alt.Chart(daily_stock_agg).mark_bar().encode(
                    x=alt.X('date:T', title='Ngày'),
                    y=alt.Y('stock_remaining:Q', title='Số Lượng Tồn Kho'),
                    tooltip=['date:T', 'stock_remaining:Q']
                ).properties(height=300)
                
                st.altair_chart(stock_revenue_chart, use_container_width=True)
                st.altair_chart(stock_quantity_chart, use_container_width=True)
        else:
            st.info(f"Không có dữ liệu {'doanh thu' if display_mode == 'Bán hàng' else 'tồn kho'} theo ngày để hiển thị trong khoảng thời gian này.")
    else:
        st.info(f"Không có dữ liệu {'bán hàng' if display_mode == 'Bán hàng' else 'tồn kho'} trong khoảng thời gian được chọn.")
else:
    st.warning("Không có dữ liệu để hiển thị.")

# ----- Phân Tích Phân Khúc Giá -----
if display_mode == "Bán hàng":
    st.subheader("💰 Phân Tích Phân Khúc Giá")
else:
    st.subheader("💰 Phân Tích Tồn Kho Theo Phân Khúc")

if not filtered_df.empty and 'segment' in filtered_df.columns:
    segment_analysis = filtered_df.groupby('segment').agg({
        'quantity_sold': 'sum',
        'stock_remaining': 'sum',
        'revenue': 'sum',
        'stock_revenue': 'sum',
        'price': 'mean'
    }).reset_index()
    
    # Thêm cột phần trăm
    if display_mode == "Bán hàng":
        total = segment_analysis['quantity_sold'].sum()
        segment_analysis['percentage'] = (segment_analysis['quantity_sold'] / total * 100).round(1)
    else:
        total = segment_analysis['stock_remaining'].sum()
        segment_analysis['percentage'] = (segment_analysis['stock_remaining'] / total * 100).round(1)
    
    # Biểu đồ Doanh thu/Doanh thu tồn kho
    if display_mode == "Bán hàng":
        st.markdown("**Doanh Thu Theo Phân Khúc**")
        revenue_chart = alt.Chart(segment_analysis).mark_bar().encode(
            x=alt.X('segment:N', title='Phân khúc', sort=['Cao', 'Trung', 'Thấp']),
            y=alt.Y('revenue:Q', title='Doanh thu (VND)'),
            color=alt.Color('segment:N', scale=alt.Scale(
                domain=['Cao', 'Trung', 'Thấp'], 
                range=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )),
            tooltip=['segment:N', 'revenue:Q', 'percentage:Q']
        ).properties(height=300)
        st.altair_chart(revenue_chart, use_container_width=True)
    else:
        st.markdown("**Doanh Thu Tồn Kho Theo Phân Khúc**")
        stock_revenue_chart = alt.Chart(segment_analysis).mark_bar().encode(
            x=alt.X('segment:N', title='Phân khúc', sort=['Cao', 'Trung', 'Thấp']),
            y=alt.Y('stock_revenue:Q', title='Doanh thu tồn kho (VND)'),
            color=alt.Color('segment:N', scale=alt.Scale(
                domain=['Cao', 'Trung', 'Thấp'], 
                range=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )),
            tooltip=['segment:N', 'stock_revenue:Q', 'percentage:Q']
        ).properties(height=300)
        st.altair_chart(stock_revenue_chart, use_container_width=True)
    
    # ----- Phân Tích Phân Khúc Giá -----
if display_mode == "Bán hàng":
    st.subheader("💰 Phân Tích Phân Khúc Giá")
else:
    st.subheader("💰 Phân Tích Tồn Kho Theo Phân Khúc")

if not filtered_df.empty and 'segment' in filtered_df.columns:
    segment_analysis = filtered_df.groupby('segment').agg({
        'quantity_sold': 'sum',
        'stock_remaining': 'sum',
        'revenue': 'sum',
        'stock_revenue': 'sum',
        'price': 'mean'
    }).reset_index()
    
    # Thêm cột phần trăm
    if display_mode == "Bán hàng":
        total = segment_analysis['revenue'].sum()
        segment_analysis['percentage'] = (segment_analysis['revenue'] / total * 100).round(1)
    else:
        total = segment_analysis['stock_revenue'].sum()
        segment_analysis['percentage'] = (segment_analysis['stock_revenue'] / total * 100).round(1)
    
    # Biểu đồ tròn Doanh thu/Doanh thu tồn kho
    if display_mode == "Bán hàng":
        st.markdown("**Doanh Thu Theo Phân Khúc**")
        base = alt.Chart(segment_analysis).encode(
            theta=alt.Theta('revenue:Q', stack=True),
            color=alt.Color('segment:N', 
                          scale=alt.Scale(domain=['Cao', 'Trung', 'Thấp'], 
                                        range=['#FF6B6B', '#4ECDC4', '#45B7D1']),
                          legend=alt.Legend(title="Phân khúc")),
            tooltip=['segment:N', 'revenue:Q', 'percentage:Q']
        )
    else:
        st.markdown("**Doanh Thu Tồn Kho Theo Phân Khúc**")
        base = alt.Chart(segment_analysis).encode(
            theta=alt.Theta('stock_revenue:Q', stack=True),
            color=alt.Color('segment:N', 
                          scale=alt.Scale(domain=['Cao', 'Trung', 'Thấp'], 
                                        range=['#FF6B6B', '#4ECDC4', '#45B7D1']),
                          legend=alt.Legend(title="Phân khúc")),
            tooltip=['segment:N', 'stock_revenue:Q', 'percentage:Q']
        )
    
    # Tạo pie chart
    pie = base.mark_arc(outerRadius=120)
    
    # Tạo text bên trong
    text_inside = base.mark_text(radius=90, size=14).encode(
        text=alt.Text('percentage:Q', format='.1f')
    )
    
    # Tạo text bên ngoài
    text_outside = base.mark_text(radius=140, size=12).encode(
        text=alt.Text('segment:N')
    )
    
    # Kết hợp tất cả
    chart = (pie + text_inside + text_outside).properties(
        width=500,
        height=400,
        title='Phân Bổ Doanh Thu Theo Phân Khúc' if display_mode == "Bán hàng" else 'Phân Bổ Doanh Thu Tồn Kho Theo Phân Khúc'
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Hiển thị bảng dữ liệu chi tiết
    st.markdown("**Chi Tiết Theo Phân Khúc**")
    if display_mode == "Bán hàng":
        st.dataframe(
            segment_analysis[['segment', 'revenue', 'percentage', 'quantity_sold', 'price']]
            .rename(columns={
                'segment': 'Phân khúc',
                'revenue': 'Doanh thu (VND)',
                'percentage': 'Tỷ lệ (%)',
                'quantity_sold': 'Số lượng bán',
                'price': 'Giá trung bình'
            }).sort_values('Phân khúc', ascending=False),
            height=None,
            width=None,
            use_container_width=True
        )
    else:
        st.dataframe(
            segment_analysis[['segment', 'stock_revenue', 'percentage', 'stock_remaining', 'price']]
            .rename(columns={
                'segment': 'Phân khúc',
                'stock_revenue': 'Doanh thu tồn kho (VND)',
                'percentage': 'Tỷ lệ (%)',
                'stock_remaining': 'Số lượng tồn kho',
                'price': 'Giá trung bình'
            }).sort_values('Phân khúc', ascending=False),
            height=None,
            width=None,
            use_container_width=True
        )
else:
    st.warning("Không có dữ liệu phân khúc để hiển thị")

# ----- Detailed Data Table -----
st.subheader("📋 Dữ Liệu Chi Tiết")
if not filtered_df.empty:
    st.info(f"Đang xem dữ liệu từ MongoDB. Thời gian hiện tại: {datetime.now().strftime('%I:%M %p +07, %d/%m/%Y')}")
    if display_mode == "Bán hàng":
        st.dataframe(
            filtered_df[['id', 'name', 'price', 'quantity_sold', 'revenue', 'segment', 'promotion', 'source_file']]
            .rename(columns={
                'id': 'ID',
                'name': 'Tên SP',
                'price': 'Giá',
                'quantity_sold': 'Số lượng bán',
                'revenue': 'Doanh thu',
                'segment': 'Phân khúc',
                'promotion': 'Khuyến mãi',
                'source_file': 'Nguồn'
            }),
            height=None,
            width=None,
            use_container_width=True
        )
    else:
        st.dataframe(
            filtered_df[['id', 'name', 'price', 'stock_remaining', 'stock_revenue', 'segment', 'promotion', 'source_file']]
            .rename(columns={
                'id': 'ID',
                'name': 'Tên SP',
                'price': 'Giá',
                'stock_remaining': 'Tồn kho',
                'stock_revenue': 'Doanh thu tồn kho',
                'segment': 'Phân khúc',
                'promotion': 'Khuyến mãi',
                'source_file': 'Nguồn'
            }),
            height=None,
            width=None,
            use_container_width=True
        )
else:
    st.warning("Không có dữ liệu để hiển thị.")