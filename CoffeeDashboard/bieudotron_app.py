import streamlit as st
import plotly.graph_objects as go

# Dữ liệu loại bao bì
packaging_types = ['Lon', 'Hộp', 'Túi', 'Gói', 'Bịch', 'Ly']
packaging_counts = [3, 23, 6, 13, 4, 1]
packaging_colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1', '#F4E04D']

# Dữ liệu loại cà phê
coffee_types = ['Hòa tan', 'Rang xay', 'Sữa']
coffee_counts = [19, 7, 24]
coffee_colors = ['#FF6F61', '#6B5B95', '#88B04B']

# Tạo dropdown trên sidebar hoặc main page
selected_option = st.selectbox(
    'Chọn biểu đồ:',
    ['Phân phối sản phẩm theo loại bao bì', 'Phân phối sản phẩm theo loại cà phê']
)

# Xử lý dữ liệu theo lựa chọn
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

# Tạo biểu đồ plotly
fig = go.Figure(
    data=[go.Pie(
        labels=labels,
        values=values,
        textinfo='label+percent',
        hoverinfo='label+value',
        marker=dict(colors=colors),
        insidetextorientation='radial'
    )]
)

fig.update_layout(
    title=title,
    showlegend=True,
    legend=dict(orientation="h", y=-0.1, x=0.3),
    margin=dict(t=60, b=40, l=20, r=20),
    height=400
)

# Hiển thị biểu đồ trong Streamlit
st.plotly_chart(fig)
