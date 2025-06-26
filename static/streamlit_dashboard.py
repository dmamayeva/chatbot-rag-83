import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown('<div class="sidebar-header">🔧 Dashboard Settings</div>', unsafe_allow_html=True)

# API URL input
default_url = "http://0.0.0.0:8000"
api_base_url = st.sidebar.text_input(
    "API Base URL", 
    value=default_url,
    help="Enter the base URL of your API (e.g., http://0.0.0.0:8000)"
)

# Auto-refresh option
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    st.rerun()

# Refresh button
if st.sidebar.button("🔄 Refresh Data", type="primary"):
    st.rerun()

# Main dashboard
st.markdown('<div class="main-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

@st.cache_data(ttl=30)  # Cache for 30 seconds
def fetch_analytics_data(api_url):
    """Fetch analytics data from the API"""
    try:
        endpoint = f"{api_url}/chat/analytics/dashboard"
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {str(e)}"

# Fetch data
with st.spinner("Fetching analytics data..."):
    data, error = fetch_analytics_data(api_base_url)

if error:
    st.error(f"❌ Error fetching data: {error}")
    st.info("Please check your API URL and ensure the endpoint is accessible.")
    st.stop()

if not data:
    st.warning("⚠️ No data received from the API")
    st.stop()

# Display raw data structure for debugging (collapsible)
with st.expander("🔍 Raw Data Structure (Debug)"):
    st.json(data)

# Process and display metrics based on common analytics structures
st.markdown("---")

# Try to extract common metrics
try:
    # Attempt to identify key metrics from the data structure
    if isinstance(data, dict):
        # Look for common metric patterns
        metrics = {}
        charts_data = {}
        
        # Extract numerical values that might be metrics
        for key, value in data.items():
            if key == 'top_queries':
                continue
            else:
                if isinstance(value, (int, float)):
                    metrics[key] = value
                elif isinstance(value, list):
                    charts_data[key] = value
                elif isinstance(value, dict):
                    # Check if it's a nested structure with metrics
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, (int, float)):
                            metrics[f"{key}_{nested_key}"] = nested_value
                        elif isinstance(nested_value, list):
                            charts_data[f"{key}_{nested_key}"] = nested_value

        # Display key metrics
        if metrics:
            st.subheader("📈 Key Metrics")
            
            # Create columns for metrics
            cols = st.columns(min(len(metrics), 4))
            for i, (metric_name, metric_value) in enumerate(metrics.items()):
                with cols[i % 4]:
                    # Format metric name for display
                    display_name = metric_name.replace('_', ' ').title()
                    
                    # Special formatting for response time metrics (convert to seconds)
                    if 'response_time' in metric_name.lower() or 'response time' in metric_name.lower():
                        value_display = f"{metric_value:.3f}s"
                        if 'avg' in metric_name.lower():
                            display_name = "Avg Response Time"
                        elif 'min' in metric_name.lower():
                            display_name = "Min Response Time"
                        elif 'max' in metric_name.lower():
                            display_name = "Max Response Time"
                    else:
                        value_display = f"{metric_value:,}" if isinstance(metric_value, int) else f"{metric_value:.2f}"
                    
                    st.metric(
                        label=display_name,
                        value=value_display
                    )

        # Display charts
        if charts_data:
            st.markdown("---")
            st.subheader("📊 Data Visualizations")
            
            for chart_name, chart_data in charts_data.items():
                if isinstance(chart_data, list) and len(chart_data) > 0:
                    st.markdown(f"**{chart_name.replace('_', ' ').title()}**")
                    
                    # Try to create appropriate charts based on data structure
                    if isinstance(chart_data[0], dict):
                        # Convert to DataFrame for easier plotting
                        try:
                            df = pd.DataFrame(chart_data)
                            
                            # Create different chart types based on columns
                            if len(df.columns) >= 2:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Bar chart
                                    if df.dtypes.iloc[1] in ['int64', 'float64']:
                                        fig_bar = px.bar(
                                            df, 
                                            x=df.columns[0], 
                                            y=df.columns[1],
                                            title=f"{chart_name} - Bar Chart"
                                        )
                                        st.plotly_chart(fig_bar, use_container_width=True)
                                
                                with col2:
                                    # Line chart if there are enough data points
                                    if len(df) > 2 and df.dtypes.iloc[1] in ['int64', 'float64']:
                                        fig_line = px.line(
                                            df, 
                                            x=df.columns[0], 
                                            y=df.columns[1],
                                            title=f"{chart_name} - Trend"
                                        )
                                        st.plotly_chart(fig_line, use_container_width=True)
                                
                                # Pie chart if appropriate
                                if len(df) <= 10 and df.dtypes.iloc[1] in ['int64', 'float64']:
                                    fig_pie = px.pie(
                                        df, 
                                        names=df.columns[0], 
                                        values=df.columns[1],
                                        title=f"{chart_name} - Distribution"
                                    )
                                    st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Display data table
                            with st.expander(f"📋 {chart_name} - Data Table"):
                                st.dataframe(df, use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"Error creating chart for {chart_name}: {str(e)}")
                            st.write("Raw data:", chart_data)
                    else:
                        # Simple list data
                        st.write(chart_data)

        # If no structured metrics found, display the data as is
        if not metrics and not charts_data:
            st.subheader("📋 API Response")
            if isinstance(data, dict):
                for key, value in data.items():
                    st.write(f"**{key}:**", value)
            else:
                st.write(data)

except Exception as e:
    st.error(f"Error processing data: {str(e)}")
    st.subheader("📋 Raw API Response")
    st.write(data)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"🔗 **API Endpoint:** {api_base_url}/chat/analytics/dashboard")
with col2:
    st.info(f"🕐 **Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
with col3:
    if auto_refresh:
        st.success("🔄 **Auto-refresh:** Enabled")
    else:
        st.info("🔄 **Auto-refresh:** Disabled")

# Help section
with st.expander("ℹ️ Help & Information"):
    st.markdown("""
    ### How to use this dashboard:
    
    1. **API URL**: Enter your API base URL in the sidebar (default: http://127.0.0.1:8000)
    2. **Auto-refresh**: Enable to automatically refresh data every 30 seconds
    3. **Manual Refresh**: Use the refresh button to update data manually
    4. **Data Structure**: Check the "Raw Data Structure" section to see what data is being received
    
    ### Expected API Response Format:
    The dashboard works best with JSON responses containing:
    - Numerical metrics (displayed as metric cards)
    - Arrays of objects (displayed as charts and tables)
    - Nested structures with metrics and data arrays
    
    ### Troubleshooting:
    - Ensure your API endpoint `/chat/analytics/dashboard` is accessible
    - Check that the API returns valid JSON
    - Verify CORS settings if running from different domains
    """)

# Auto-refresh mechanism
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()