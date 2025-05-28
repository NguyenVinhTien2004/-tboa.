import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import altair as alt
from datetime import datetime, date
from ast import literal_eval

# ----- Load Data from Multiple Excel Files -----
@st.cache_data
def load_data(uploaded_files):
    all_data = pd.DataFrame()
    all_dates = []
    
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_excel(uploaded_file)
            product_data = []
            
            for _, row in df.iterrows():
                try:
                    sales_history = literal_eval(row['sales_history']) if pd.notna(row['sales_history']) else []
                    stock_history = literal_eval(row['stock_history']) if pd.notna(row['stock_history']) else []
                    price_history = literal_eval(row['price_history']) if pd.notna(row['price_history']) else []
                    
                    stock_decreased_values = [float(entry.get('stock_decreased', 0)) for entry in stock_history if isinstance(entry, dict)]
                    has_meaningful_sales = any(abs(val) > 1e-10 for val in stock_decreased_values)
                    has_valid_data = True
                    for val in stock_decreased_values:
                        if abs(val) < 1e-10:
                            continue
                        if abs(val - round(val)) > 1e-10:
                            has_valid_data = False
                            break
                    
                    if not has_meaningful_sales or not has_valid_data:
                        continue
                    
                    total_sold = sum(float(entry.get('stock_decreased', 0)) for entry in stock_history if isinstance(entry, dict))
                    current_price = float(row['price']) if pd.notna(row['price']) else 0
                    
                    for entry in stock_history:
                        if isinstance(entry, dict) and 'date' in entry and entry['date']:
                            try:
                                entry_date = datetime.strptime(entry['date'], '%Y-%m-%d').date()
                                all_dates.append(entry_date)
                            except (ValueError, TypeError):
                                continue
                    
                    product_data.append({
                        'id': row['id'],
                        'name': row['name'],
                        'category': row['category'],
                        'price': current_price,
                        'promotion': row['promotion'],
                        'total_sold': total_sold,
                        'revenue': current_price * total_sold,
                        'sales_history': sales_history,
                        'stock_history': stock_history,
                        'price_history': price_history,
                        'source_file': uploaded_file.name
                    })
                except Exception as e:
                    st.error(f"Error processing row {_} in file {uploaded_file.name}: {str(e)}")
                    continue
            
            if product_data:
                file_df = pd.DataFrame(product_data)
                all_data = pd.concat([all_data, file_df], ignore_index=True)
        except Exception as e:
            st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
            continue
    
    if not all_dates:
        st.warning("No valid dates found in stock history. Using default date range.")
        min_date = date(2025, 3, 5)
        max_date = date(2025, 5, 25)
    else:
        min_date = min(all_dates)
        max_date = max(all_dates)
    
    return all_data, min_date, max_date

def apply_clustering(df):
    if len(df) > 0:
        prices = df[['price']].dropna()
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
            st.warning("Not enough data for clustering. Assigning 'Thấp' to all.")
            df['segment'] = 'Thấp'
    else:
        df['segment'] = None
    return df

def filter_by_date_range(df, start_date, end_date):
    filtered_data = []
    for _, row in df.iterrows():
        try:
            total_sold = 0
            total_stock_increased = 0
            for entry in row['stock_history']:
                try:
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
            st.error(f"Error filtering row {_}: {str(e)}")
            continue
    return pd.DataFrame(filtered_data)

# ----- Main Dashboard -----
st.set_page_config(layout="wide", page_title="Dashboard Phân Khúc Doanh Thu và Tồn Kho - KingFoodMart")

st.sidebar.header("Tải lên dữ liệu")
uploaded_files = st.sidebar.file_uploader("Chọn file Excel", type=["xlsx", "xls"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Vui lòng tải lên ít nhất một file Excel để bắt đầu.")
    st.stop()

df, min_date, max_date = load_data(uploaded_files)

if df.empty:
    st.error("The DataFrame is empty after loading. Please check the data source.")
    st.stop()

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
    
    col1, col2 = st.columns(2)
    
    with col1:
        if display_mode == "Bán hàng":
            st.markdown("**Doanh Thu Theo Phân Khúc**")
            segment_chart = alt.Chart(segment_analysis).mark_arc(innerRadius=50).encode(
                theta=alt.Theta('revenue:Q', stack=True),
                color=alt.Color('segment:N', scale=alt.Scale(domain=['Cao', 'Trung', 'Thấp'], range=['#FF6B6B', '#4ECDC4', '#45B7D1'])),
                tooltip=['segment:N', 'revenue:Q']
            ).properties(width=300, height=300)
            st.altair_chart(segment_chart, use_container_width=True)
        else:
            st.markdown("**Doanh Thu Tồn Kho Theo Phân Khúc**")
            segment_stock_chart = alt.Chart(segment_analysis).mark_arc(innerRadius=50).encode(
                theta=alt.Theta('stock_revenue:Q', stack=True),
                color=alt.Color('segment:N', scale=alt.Scale(domain=['Cao', 'Trung', 'Thấp'], range=['#FF6B6B', '#4ECDC4', '#45B7D1'])),
                tooltip=['segment:N', 'stock_revenue:Q']
            ).properties(width=300, height=300)
            st.altair_chart(segment_stock_chart, use_container_width=True)
    
    with col2:
        if display_mode == "Bán hàng":
            st.markdown("**Số Lượng Bán Theo Phân Khúc**")
            quantity_chart = alt.Chart(segment_analysis).mark_arc(innerRadius=50).encode(
                theta=alt.Theta('quantity_sold:Q', stack=True),
                color=alt.Color('segment:N', scale=alt.Scale(domain=['Cao', 'Trung', 'Thấp'], range=['#FF6B6B', '#4ECDC4', '#45B7D1'])),
                tooltip=['segment:N', 'quantity_sold:Q']
            ).properties(width=300, height=300)
            st.altair_chart(quantity_chart, use_container_width=True)
        else:
            st.markdown("**Số Lượng Tồn Kho Theo Phân Khúc**")
            stock_quantity_chart = alt.Chart(segment_analysis).mark_arc(innerRadius=50).encode(
                theta=alt.Theta('stock_remaining:Q', stack=True),
                color=alt.Color('segment:N', scale=alt.Scale(domain=['Cao', 'Trung', 'Thấp'], range=['#FF6B6B', '#4ECDC4', '#45B7D1'])),
                tooltip=['segment:N', 'stock_remaining:Q']
            ).properties(width=300, height=300)
            st.altair_chart(stock_quantity_chart, use_container_width=True)
else:
    st.warning("Không có dữ liệu phân khúc để hiển thị")

# ----- Detailed Data Table -----
st.subheader("📋 Dữ Liệu Chi Tiết")
if not filtered_df.empty:
    st.info(f"Đang xem dữ liệu từ {len(filtered_df['source_file'].unique())} file. Thời gian hiện tại: {datetime.now().strftime('%I:%M %p +07, %d/%m/%Y')}")
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