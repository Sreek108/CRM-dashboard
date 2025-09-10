import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="QUARA FINANCE - Complete CRM Dashboard", 
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
    .main { padding-top: 1rem; }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    
    /* User role badge */
    .role-badge {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    /* Status indicators */
    .status-high { color: #dc3545; font-weight: bold; }
    .status-medium { color: #ffc107; font-weight: bold; }
    .status-low { color: #28a745; font-weight: bold; }
    .status-available { color: #28a745; }
    .status-busy { color: #dc3545; }
    .status-offline { color: #6c757d; }
    
    /* Section headers */
    .section-header {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    /* KPI styling */
    .kpi-container {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    /* Alert boxes */
    .alert-success { background: #d4edda; border: 1px solid #c3e6cb; padding: 1rem; border-radius: 5px; color: #155724; }
    .alert-warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 5px; color: #856404; }
    .alert-danger { background: #f8d7da; border: 1px solid #f1aeb5; padding: 1rem; border-radius: 5px; color: #721c24; }
    
    /* Table styling */
    .dataframe { font-size: 0.9rem; }
    
    /* Sidebar styling */
    .sidebar .sidebar-content { background: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'Agent'
if 'selected_agent' not in st.session_state:
    st.session_state.selected_agent = 1

# Data loading functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load all dashboard data"""
    try:
        # In a real deployment, you'd load from your database
        # For demo, we'll generate sample data
        return generate_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return generate_sample_data()

def generate_sample_data():
    """Generate comprehensive sample data"""
    np.random.seed(42)
    
    # Lead data
    lead_statuses = ['New', 'In Progress', 'Interested', 'Closed - Won', 'Closed - Lost']
    countries = ['Saudi Arabia', 'UAE', 'India', 'Kuwait', 'Bahrain']
    
    leads_data = []
    for i in range(200):
        status = np.random.choice(lead_statuses, p=[0.25, 0.20, 0.15, 0.25, 0.15])
        country = np.random.choice(countries)
        created_date = datetime.now() - timedelta(days=np.random.randint(1, 180))
        
        leads_data.append({
            'lead_id': f'LEAD_{i+1:04d}',
            'client_name': f'Client {i+1}',
            'company': f'Company {chr(65 + i%5)}{i+1}',
            'country': country,
            'status': status,
            'agent_id': np.random.randint(1, 11),
            'deal_value': np.random.randint(50000, 500000),
            'created_date': created_date,
            'probability': np.random.uniform(0.1, 0.9),
            'priority': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2])
        })
    
    leads_df = pd.DataFrame(leads_data)
    
    # Call data
    calls_data = []
    call_outcomes = ['Completed', 'No Answer', 'Busy', 'Voicemail']
    sentiments = ['Positive', 'Neutral', 'Negative']
    
    for i in range(500):
        call_date = datetime.now() - timedelta(days=np.random.randint(1, 30))
        duration = np.random.randint(30, 1800) if np.random.random() > 0.3 else 0
        
        calls_data.append({
            'call_id': f'CALL_{i+1:04d}',
            'agent_id': np.random.randint(1, 11),
            'call_date': call_date,
            'duration': duration,
            'outcome': np.random.choice(call_outcomes, p=[0.45, 0.25, 0.2, 0.1]),
            'sentiment': np.random.choice(sentiments, p=[0.5, 0.35, 0.15]),
            'lead_converted': np.random.choice([True, False], p=[0.3, 0.7])
        })
    
    calls_df = pd.DataFrame(calls_data)
    
    # Agent data
    agents_data = []
    for i in range(1, 11):
        agent_calls = calls_df[calls_df['agent_id'] == i]
        agent_leads = leads_df[leads_df['agent_id'] == i]
        
        total_calls = len(agent_calls)
        completed_calls = len(agent_calls[agent_calls['outcome'] == 'Completed'])
        
        agents_data.append({
            'agent_id': i,
            'agent_name': f'Agent {i}',
            'total_calls_today': np.random.randint(8, 25),
            'total_calls_week': np.random.randint(40, 100),
            'total_calls_month': total_calls,
            'success_rate': completed_calls / total_calls if total_calls > 0 else 0,
            'leads_converted': len(agent_leads[agent_leads['status'] == 'Closed - Won']),
            'revenue_generated': agent_leads[agent_leads['status'] == 'Closed - Won']['deal_value'].sum(),
            'availability': np.random.choice(['Available', 'Busy', 'In Meeting', 'Offline'], p=[0.4, 0.3, 0.2, 0.1]),
            'customer_satisfaction': np.random.uniform(3.5, 5.0)
        })
    
    agents_df = pd.DataFrame(agents_data)
    
    # Task data
    tasks_data = []
    task_types = ['Follow-up Call', 'Send Proposal', 'Schedule Meeting', 'Client Visit']
    
    for i in range(100):
        due_date = datetime.now() + timedelta(days=np.random.randint(-5, 15))
        is_overdue = due_date < datetime.now()
        
        tasks_data.append({
            'task_id': f'TASK_{i+1:04d}',
            'agent_id': np.random.randint(1, 11),
            'task_type': np.random.choice(task_types),
            'priority': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2]),
            'due_date': due_date,
            'status': 'Overdue' if is_overdue else np.random.choice(['Pending', 'In Progress', 'Completed'], p=[0.5, 0.3, 0.2]),
            'client_name': f'Client {np.random.randint(1, 200)}'
        })
    
    tasks_df = pd.DataFrame(tasks_data)
    
    # Geographic data
    geo_data = []
    for country in countries:
        country_leads = leads_df[leads_df['country'] == country]
        geo_data.append({
            'country': country,
            'total_leads': len(country_leads),
            'won_leads': len(country_leads[country_leads['status'] == 'Closed - Won']),
            'conversion_rate': len(country_leads[country_leads['status'] == 'Closed - Won']) / len(country_leads) if len(country_leads) > 0 else 0,
            'total_revenue': country_leads[country_leads['status'] == 'Closed - Won']['deal_value'].sum(),
            'response_rate': np.random.uniform(0.6, 0.9)
        })
    
    geo_df = pd.DataFrame(geo_data)
    
    return {
        'leads': leads_df,
        'calls': calls_df,
        'agents': agents_df,
        'tasks': tasks_df,
        'geographic': geo_df
    }

def create_availability_heatmap(agents_df):
    """Create agent availability heatmap"""
    # Create hourly availability data for agents
    hours = list(range(9, 18))  # 9 AM to 5 PM
    agents = [f'Agent {i}' for i in range(1, 11)]
    
    # Generate realistic availability data
    availability_matrix = []
    for agent in agents:
        agent_schedule = []
        for hour in hours:
            # Higher availability in morning, lower in afternoon
            base_prob = 0.9 if hour < 12 else 0.7 if hour < 15 else 0.5
            availability = np.random.choice([1, 0.5, 0], p=[base_prob, (1-base_prob)*0.7, (1-base_prob)*0.3])
            agent_schedule.append(availability)
        availability_matrix.append(agent_schedule)
    
    # Create heatmap
    fig = px.imshow(
        availability_matrix,
        x=[f'{hour:02d}:00' for hour in hours],
        y=agents,
        color_continuous_scale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
        title='🕐 Real-Time Agent Availability Heatmap',
        labels={'x': 'Time', 'y': 'Agent', 'color': 'Availability'}
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_lead_funnel(leads_df):
    """Create lead conversion funnel"""
    funnel_data = leads_df['status'].value_counts()
    
    fig = go.Figure(go.Funnel(
        y=funnel_data.index,
        x=funnel_data.values,
        textinfo="value+percent initial",
        marker_color=["#3498db", "#e74c3c", "#f39c12", "#27ae60", "#8e44ad"]
    ))
    
    fig.update_layout(title="📊 Lead Conversion Funnel", height=500)
    return fig

def create_call_activity_chart(calls_df):
    """Create call activity dashboard"""
    # Daily call activity for last 30 days
    calls_df['date'] = pd.to_datetime(calls_df['call_date']).dt.date
    daily_calls = calls_df.groupby('date').agg({
        'call_id': 'count',
        'outcome': lambda x: (x == 'Completed').sum()
    }).rename(columns={'call_id': 'total_calls', 'outcome': 'successful_calls'})
    
    daily_calls['success_rate'] = daily_calls['successful_calls'] / daily_calls['total_calls']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Call Volume', 'Success Rate Trend'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Call volume
    fig.add_trace(
        go.Scatter(x=daily_calls.index, y=daily_calls['total_calls'], 
                  mode='lines+markers', name='Total Calls', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=daily_calls.index, y=daily_calls['successful_calls'], 
                  mode='lines+markers', name='Successful Calls', line=dict(color='green')),
        row=1, col=1
    )
    
    # Success rate
    fig.add_trace(
        go.Scatter(x=daily_calls.index, y=daily_calls['success_rate'], 
                  mode='lines+markers', name='Success Rate', line=dict(color='orange')),
        row=2, col=1
    )
    
    fig.update_layout(height=600, title_text="📞 AI Call Activity Dashboard")
    return fig

def create_geographic_map(geo_df):
    """Create geographic performance map"""
    # Create world map with country performance
    fig = px.choropleth(
        geo_df,
        locations='country',
        locationmode='country names',
        color='conversion_rate',
        hover_data=['total_leads', 'total_revenue'],
        color_continuous_scale='Viridis',
        title='🌍 Global Performance Map'
    )
    
    fig.update_layout(height=500)
    return fig

def create_revenue_forecast():
    """Create ML-based revenue forecasting"""
    # Generate historical and predicted revenue data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now() + timedelta(days=14), freq='D')
    
    revenue_data = []
    for i, date in enumerate(dates):
        if date <= datetime.now():
            # Historical data with some trend
            base_revenue = 150000 + np.sin(i * 0.1) * 20000
            revenue = base_revenue + np.random.normal(0, 10000)
            data_type = 'Historical'
        else:
            # Predicted data
            base_revenue = 160000 + np.sin(i * 0.1) * 25000
            revenue = base_revenue + np.random.normal(0, 15000)
            data_type = 'Predicted'
        
        revenue_data.append({
            'date': date,
            'revenue': max(0, revenue),
            'type': data_type
        })
    
    forecast_df = pd.DataFrame(revenue_data)
    
    fig = px.line(forecast_df, x='date', y='revenue', color='type',
                  title='🔮 ML Revenue Forecasting (14-Day Prediction)')
    
    fig.update_layout(height=400)
    return fig

# Main dashboard function
def main():
    """Main dashboard application"""
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1>🏢 QUARA FINANCE</h1>
        <h2>Complete CRM Dashboard with AI Predictions</h2>
        <p>Multi-Level Call Quality Management & Lead Analytics System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 User Access Control")
        
        # User role selection
        user_role = st.selectbox(
            "Select your access level:",
            ["Agent", "Team Lead", "Manager", "Executive"],
            index=["Agent", "Team Lead", "Manager", "Executive"].index(st.session_state.user_role)
        )
        st.session_state.user_role = user_role
        
        st.markdown(f'<div class="role-badge">{user_role}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dashboard navigation
        if user_role == "Agent":
            dashboards = [
                "🏠 My Overview",
                "📊 My Performance", 
                "✅ My Tasks",
                "📞 My Calls"
            ]
        elif user_role == "Team Lead":
            dashboards = [
                "🏠 Team Overview",
                "👥 Agent Performance",
                "✅ Task Management", 
                "🕐 Availability Heatmap",
                "📞 Call Analytics"
            ]
        elif user_role == "Manager":
            dashboards = [
                "🏠 Executive Summary",
                "📊 Lead Status Dashboard",
                "📞 AI Call Activity",
                "🌍 Geographic Dashboard",
                "📈 Conversion Analytics"
            ]
        else:  # Executive
            dashboards = [
                "🏠 Strategic Overview",
                "📊 Complete Lead Analytics", 
                "📞 AI Call Intelligence",
                "✅ Task & Follow-up Center",
                "🕐 Agent Availability Center",
                "📈 Conversion Dashboard",
                "🌍 Geographic Intelligence",
                "🔮 ML Predictions & Forecasting"
            ]
        
        selected_dashboard = st.selectbox("📋 Select Dashboard:", dashboards)
        
        # Agent selection for relevant roles
        if user_role in ["Agent", "Team Lead"] and "Agent" in selected_dashboard:
            st.markdown("---")
            st.session_state.selected_agent = st.selectbox(
                "Select Agent:", 
                range(1, 11), 
                format_func=lambda x: f"Agent {x}",
                index=st.session_state.selected_agent - 1
            )
        
        st.markdown("---")
        
        # Quick stats in sidebar
        st.markdown("### 📈 Quick Stats")
        total_leads = len(data['leads'])
        won_leads = len(data['leads'][data['leads']['status'] == 'Closed - Won'])
        total_calls_today = data['agents']['total_calls_today'].sum()
        
        st.metric("Total Leads", total_leads)
        st.metric("Won Today", won_leads, delta="12")
        st.metric("Calls Today", total_calls_today, delta="45")
        
        st.markdown("---")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content area
    if selected_dashboard == "🏠 Strategic Overview" or (user_role == "Executive" and "Strategic" in selected_dashboard):
        show_strategic_overview(data)
    elif "Lead Status" in selected_dashboard or "Complete Lead" in selected_dashboard:
        show_lead_status_dashboard(data)
    elif "AI Call" in selected_dashboard:
        show_call_activity_dashboard(data)
    elif "Task" in selected_dashboard:
        show_task_dashboard(data)
    elif "Availability" in selected_dashboard:
        show_availability_dashboard(data)
    elif "Conversion" in selected_dashboard:
        show_conversion_dashboard(data)
    elif "Geographic" in selected_dashboard:
        show_geographic_dashboard(data)
    elif "ML Predictions" in selected_dashboard or "Forecasting" in selected_dashboard:
        show_ml_forecasting_dashboard(data)
    elif "My Performance" in selected_dashboard:
        show_agent_performance(data, st.session_state.selected_agent)
    elif "Team Overview" in selected_dashboard:
        show_team_overview(data)
    elif "Agent Performance" in selected_dashboard:
        show_agent_management(data)
    else:
        show_overview_dashboard(data, user_role)

def show_strategic_overview(data):
    """Executive strategic overview dashboard"""
    st.markdown('<div class="section-header"><h2>🎯 Strategic Overview Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Top KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_leads = len(data['leads'])
    won_leads = len(data['leads'][data['leads']['status'] == 'Closed - Won'])
    total_revenue = data['leads'][data['leads']['status'] == 'Closed - Won']['deal_value'].sum()
    total_calls = data['agents']['total_calls_month'].sum()
    avg_success_rate = data['agents']['success_rate'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="kpi-container">
            <h3>{total_leads}</h3>
            <p>Total Leads</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-container">
            <h3>{won_leads}</h3>
            <p>Won Deals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-container">
            <h3>SAR {total_revenue:,.0f}</h3>
            <p>Total Revenue</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-container">
            <h3>{total_calls}</h3>
            <p>Total Calls</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="kpi-container">
            <h3>{avg_success_rate:.1%}</h3>
            <p>Avg Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main dashboard components
    col1, col2 = st.columns(2)
    
    with col1:
        # Lead funnel
        funnel_fig = create_lead_funnel(data['leads'])
        st.plotly_chart(funnel_fig, use_container_width=True)
    
    with col2:
        # Revenue forecast
        forecast_fig = create_revenue_forecast()
        st.plotly_chart(forecast_fig, use_container_width=True)
    
    # Geographic performance
    geo_fig = create_geographic_map(data['geographic'])
    st.plotly_chart(geo_fig, use_container_width=True)
    
    # Agent performance summary
    st.markdown("### 👥 Agent Performance Summary")
    
    agent_summary = data['agents'][['agent_name', 'total_calls_month', 'success_rate', 'revenue_generated', 'availability']].copy()
    agent_summary['success_rate'] = agent_summary['success_rate'].apply(lambda x: f"{x:.1%}")
    agent_summary['revenue_generated'] = agent_summary['revenue_generated'].apply(lambda x: f"SAR {x:,.0f}")
    
    st.dataframe(agent_summary, use_container_width=True)

def show_lead_status_dashboard(data):
    """Complete lead status analytics dashboard"""
    st.markdown('<div class="section-header"><h2>📊 Lead Status Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Lead status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    status_counts = data['leads']['status'].value_counts()
    
    with col1:
        st.metric("New Leads", status_counts.get('New', 0), delta="8")
    with col2:
        st.metric("In Progress", status_counts.get('In Progress', 0), delta="3")  
    with col3:
        st.metric("Interested", status_counts.get('Interested', 0), delta="5")
    with col4:
        st.metric("Closed Won", status_counts.get('Closed - Won', 0), delta="12")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lead status pie chart
        fig_pie = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="📈 Lead Status Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Lead trend over time
        leads_by_date = data['leads'].groupby(data['leads']['created_date'].dt.date).size()
        fig_trend = px.line(x=leads_by_date.index, y=leads_by_date.values, title="📅 Lead Generation Trend")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Lead details table
    st.markdown("### 📋 Lead Details")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.multiselect("Filter by Status", data['leads']['status'].unique(), default=data['leads']['status'].unique())
    with col2:
        country_filter = st.multiselect("Filter by Country", data['leads']['country'].unique(), default=data['leads']['country'].unique())
    with col3:
        priority_filter = st.multiselect("Filter by Priority", data['leads']['priority'].unique(), default=data['leads']['priority'].unique())
    
    # Filter data
    filtered_leads = data['leads'][
        (data['leads']['status'].isin(status_filter)) &
        (data['leads']['country'].isin(country_filter)) &
        (data['leads']['priority'].isin(priority_filter))
    ]
    
    st.dataframe(filtered_leads[['lead_id', 'client_name', 'company', 'country', 'status', 'deal_value', 'priority']], use_container_width=True)

def show_call_activity_dashboard(data):
    """AI-powered call activity dashboard"""
    st.markdown('<div class="section-header"><h2>📞 AI Call Activity Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Call metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_calls = len(data['calls'])
    completed_calls = len(data['calls'][data['calls']['outcome'] == 'Completed'])
    success_rate = completed_calls / total_calls if total_calls > 0 else 0
    avg_duration = data['calls']['duration'].mean()
    
    with col1:
        st.metric("Total Calls", total_calls, delta="23")
    with col2:
        st.metric("Completed Calls", completed_calls, delta="15")
    with col3:
        st.metric("Success Rate", f"{success_rate:.1%}", delta="3.2%")
    with col4:
        st.metric("Avg Duration", f"{avg_duration/60:.1f} min", delta="1.2 min")
    
    # Call activity chart
    call_activity_fig = create_call_activity_chart(data['calls'])
    st.plotly_chart(call_activity_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Call outcome distribution
        outcome_counts = data['calls']['outcome'].value_counts()
        fig_outcome = px.bar(x=outcome_counts.index, y=outcome_counts.values, 
                           title="📊 Call Outcome Distribution")
        st.plotly_chart(fig_outcome, use_container_width=True)
    
    with col2:
        # Call sentiment analysis
        sentiment_counts = data['calls']['sentiment'].value_counts()
        fig_sentiment = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                             title="😊 Call Sentiment Analysis")
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # AI Insights
    st.markdown("### 🤖 AI-Generated Insights")
    
    insights = [
        "📈 Call volume increased by 23% compared to last week",
        "🕐 Peak performance window: 10:00-11:00 AM (highest success rate)",
        "😊 Positive sentiment calls have 67% higher conversion rate",
        "📞 Follow-up calls within 24 hours show 45% better success rate",
        "🎯 Agent 3 and Agent 7 consistently outperform on call quality"
    ]
    
    for insight in insights:
        st.info(insight)

def show_task_dashboard(data):
    """Task and follow-up management dashboard"""
    st.markdown('<div class="section-header"><h2>✅ Follow-up & Task Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Task metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_tasks = len(data['tasks'])
    pending_tasks = len(data['tasks'][data['tasks']['status'] == 'Pending'])
    overdue_tasks = len(data['tasks'][data['tasks']['status'] == 'Overdue'])
    completed_tasks = len(data['tasks'][data['tasks']['status'] == 'Completed'])
    
    with col1:
        st.metric("Total Tasks", total_tasks)
    with col2:
        st.metric("Pending", pending_tasks, delta="-5")
    with col3:
        st.metric("Overdue", overdue_tasks, delta="2", delta_color="inverse")
    with col4:
        st.metric("Completed", completed_tasks, delta="8")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Task status distribution
        status_counts = data['tasks']['status'].value_counts()
        fig_status = px.pie(values=status_counts.values, names=status_counts.index,
                           title="📋 Task Status Distribution")
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        # Task priority breakdown
        priority_counts = data['tasks']['priority'].value_counts()
        fig_priority = px.bar(x=priority_counts.index, y=priority_counts.values,
                            title="⚡ Task Priority Breakdown",
                            color=priority_counts.index,
                            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
        st.plotly_chart(fig_priority, use_container_width=True)
    
    # Upcoming tasks table
    st.markdown("### 📅 Upcoming Tasks")
    
    upcoming_tasks = data['tasks'][data['tasks']['due_date'] >= datetime.now()].sort_values('due_date').head(10)
    
    # Style the table based on priority
    def style_priority(val):
        if val == 'High':
            return 'background-color: #ffebee; color: #c62828;'
        elif val == 'Medium':
            return 'background-color: #fff8e1; color: #f57f17;'
        else:
            return 'background-color: #e8f5e8; color: #2e7d32;'
    
    styled_tasks = upcoming_tasks[['task_id', 'task_type', 'priority', 'due_date', 'status', 'client_name']].style.applymap(style_priority, subset=['priority'])
    st.dataframe(styled_tasks, use_container_width=True)
    
    # Overdue alerts
    if overdue_tasks > 0:
        st.markdown(f"""
        <div class="alert-danger">
            <strong>⚠️ Alert:</strong> You have {overdue_tasks} overdue tasks that require immediate attention!
        </div>
        """, unsafe_allow_html=True)

def show_availability_dashboard(data):
    """Agent availability heatmap dashboard"""
    st.markdown('<div class="section-header"><h2>🕐 Agent Availability Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Current availability status
    col1, col2, col3, col4 = st.columns(4)
    
    availability_counts = data['agents']['availability'].value_counts()
    
    with col1:
        st.metric("Available", availability_counts.get('Available', 0), delta="0")
    with col2:
        st.metric("Busy", availability_counts.get('Busy', 0), delta="1")
    with col3:
        st.metric("In Meeting", availability_counts.get('In Meeting', 0), delta="-1")
    with col4:
        st.metric("Offline", availability_counts.get('Offline', 0), delta="0")
    
    # Availability heatmap
    availability_fig = create_availability_heatmap(data['agents'])
    st.plotly_chart(availability_fig, use_container_width=True)
    
    # Current agent status
    st.markdown("### 👥 Current Agent Status")
    
    agent_status = data['agents'][['agent_name', 'availability', 'total_calls_today', 'customer_satisfaction']].copy()
    
    # Color code availability
    def color_availability(val):
        colors = {
            'Available': 'background-color: #d4edda; color: #155724;',
            'Busy': 'background-color: #f8d7da; color: #721c24;',
            'In Meeting': 'background-color: #fff3cd; color: #856404;',
            'Offline': 'background-color: #e2e3e5; color: #383d41;'
        }
        return colors.get(val, '')
    
    styled_status = agent_status.style.applymap(color_availability, subset=['availability'])
    st.dataframe(styled_status, use_container_width=True)
    
    # Availability insights
    st.markdown("### 💡 Availability Insights")
    st.info("📊 Peak availability window: 9:00-11:00 AM (90% agents available)")
    st.info("⚡ Recommended meeting times: 10:00 AM or 2:00 PM")
    st.warning("⚠️ Low availability period: 4:00-5:00 PM (only 50% available)")

def show_conversion_dashboard(data):
    """Lead conversion analytics dashboard"""
    st.markdown('<div class="section-header"><h2>📈 Conversion Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Conversion metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_leads = len(data['leads'])
    won_leads = len(data['leads'][data['leads']['status'] == 'Closed - Won'])
    lost_leads = len(data['leads'][data['leads']['status'] == 'Closed - Lost'])
    conversion_rate = won_leads / total_leads if total_leads > 0 else 0
    
    with col1:
        st.metric("Conversion Rate", f"{conversion_rate:.1%}", delta="2.3%")
    with col2:
        st.metric("Won Deals", won_leads, delta="12")
    with col3:
        st.metric("Lost Deals", lost_leads, delta="-3")
    with col4:
        st.metric("Win Rate", f"{won_leads/(won_leads+lost_leads):.1%}" if (won_leads+lost_leads) > 0 else "0%", delta="5.1%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Conversion funnel
        funnel_fig = create_lead_funnel(data['leads'])
        st.plotly_chart(funnel_fig, use_container_width=True)
    
    with col2:
        # Revenue by status
        revenue_by_status = data['leads'].groupby('status')['deal_value'].sum().sort_values(ascending=False)
        fig_revenue = px.bar(x=revenue_by_status.index, y=revenue_by_status.values,
                           title="💰 Revenue by Lead Status")
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Conversion analysis by agent
    st.markdown("### 👥 Agent Conversion Performance")
    
    agent_conversion = data['leads'].groupby('agent_id').agg({
        'lead_id': 'count',
        'status': lambda x: (x == 'Closed - Won').sum(),
        'deal_value': lambda x: x[x.index[data['leads'].loc[x.index, 'status'] == 'Closed - Won']].sum() if len(x[x.index[data['leads'].loc[x.index, 'status'] == 'Closed - Won']]) > 0 else 0
    }).rename(columns={'lead_id': 'total_leads', 'status': 'won_leads', 'deal_value': 'revenue'})
    
    agent_conversion['conversion_rate'] = agent_conversion['won_leads'] / agent_conversion['total_leads']
    agent_conversion['agent_name'] = [f'Agent {i}' for i in agent_conversion.index]
    
    fig_agent_conversion = px.scatter(agent_conversion, x='total_leads', y='conversion_rate', 
                                    size='revenue', hover_name='agent_name',
                                    title="🎯 Agent Performance: Leads vs Conversion Rate (Bubble = Revenue)")
    st.plotly_chart(fig_agent_conversion, use_container_width=True)

def show_geographic_dashboard(data):
    """Geographic performance analysis dashboard"""
    st.markdown('<div class="section-header"><h2>🌍 Geographic Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Geographic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_countries = len(data['geographic'])
    best_country = data['geographic'].loc[data['geographic']['conversion_rate'].idxmax(), 'country']
    total_geo_revenue = data['geographic']['total_revenue'].sum()
    avg_response_rate = data['geographic']['response_rate'].mean()
    
    with col1:
        st.metric("Active Countries", total_countries)
    with col2:
        st.metric("Best Performing", best_country)
    with col3:
        st.metric("Total Revenue", f"SAR {total_geo_revenue:,.0f}")
    with col4:
        st.metric("Avg Response Rate", f"{avg_response_rate:.1%}")
    
    # Geographic visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Country performance bar chart
        fig_countries = px.bar(data['geographic'], x='country', y='conversion_rate',
                             title="🏆 Conversion Rate by Country",
                             color='conversion_rate',
                             color_continuous_scale='Viridis')
        st.plotly_chart(fig_countries, use_container_width=True)
    
    with col2:
        # Revenue distribution
        fig_revenue = px.pie(data['geographic'], values='total_revenue', names='country',
                           title="💰 Revenue Distribution by Country")
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Geographic performance table
    st.markdown("### 📊 Detailed Geographic Performance")
    
    geo_table = data['geographic'].copy()
    geo_table['conversion_rate'] = geo_table['conversion_rate'].apply(lambda x: f"{x:.1%}")
    geo_table['response_rate'] = geo_table['response_rate'].apply(lambda x: f"{x:.1%}")
    geo_table['total_revenue'] = geo_table['total_revenue'].apply(lambda x: f"SAR {x:,.0f}")
    
    st.dataframe(geo_table, use_container_width=True)
    
    # Geographic insights
    st.markdown("### 🎯 Geographic Insights")
    
    best_conversion = data['geographic'].loc[data['geographic']['conversion_rate'].idxmax()]
    best_revenue = data['geographic'].loc[data['geographic']['total_revenue'].idxmax()]
    
    st.success(f"🏆 **Best Conversion Rate:** {best_conversion['country']} ({best_conversion['conversion_rate']:.1%})")
    st.info(f"💰 **Highest Revenue:** {best_revenue['country']} (SAR {best_revenue['total_revenue']:,.0f})")
    st.warning(f"📈 **Growth Opportunity:** Focus on improving response rates in underperforming regions")

def show_ml_forecasting_dashboard(data):
    """ML predictions and forecasting dashboard"""
    st.markdown('<div class="section-header"><h2>🔮 ML Predictions & Forecasting</h2></div>', unsafe_allow_html=True)
    
    # Prediction metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Generate predictions (in real app, this would use trained models)
    predicted_leads_7d = np.random.randint(45, 65)
    predicted_revenue_7d = np.random.randint(800000, 1200000)
    confidence_score = np.random.uniform(0.85, 0.95)
    trend_direction = "Upward"
    
    with col1:
        st.metric("7-Day Lead Forecast", predicted_leads_7d, delta="8")
    with col2:
        st.metric("7-Day Revenue Forecast", f"SAR {predicted_revenue_7d:,.0f}", delta="12.5%")
    with col3:
        st.metric("Model Confidence", f"{confidence_score:.1%}", delta="2.1%")
    with col4:
        st.metric("Trend Direction", trend_direction, delta="Positive")
    
    # Revenue forecasting chart
    forecast_fig = create_revenue_forecast()
    st.plotly_chart(forecast_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lead prediction model
        dates = pd.date_range(start=datetime.now(), periods=14, freq='D')
        predicted_leads = [np.random.randint(8, 15) for _ in dates]
        
        fig_leads = px.line(x=dates, y=predicted_leads, title="📈 14-Day Lead Generation Forecast")
        fig_leads.add_scatter(x=dates, y=[np.random.randint(6, 12) for _ in dates], 
                            mode='lines', name='Conservative Forecast', line=dict(dash='dash'))
        st.plotly_chart(fig_leads, use_container_width=True)
    
    with col2:
        # Success rate prediction
        predicted_success_rates = [np.random.uniform(0.65, 0.85) for _ in dates]
        
        fig_success = px.line(x=dates, y=predicted_success_rates, title="🎯 Success Rate Forecast")
        fig_success.update_yaxis(tickformat='.1%')
        st.plotly_chart(fig_success, use_container_width=True)
    
    # AI Insights and Recommendations
    st.markdown("### 🧠 AI-Generated Insights & Recommendations")
    
    insights = [
        ("📈", "Lead Generation", "Expected 15% increase in lead volume next week based on seasonal patterns"),
        ("🎯", "Success Rate", "Optimal calling window identified: 10:00-11:00 AM shows 23% higher success rate"),
        ("💰", "Revenue Opportunity", "UAE market showing strongest growth potential with 18% projected increase"),
        ("⚡", "Process Optimization", "Automated follow-up system could improve conversion by 12%"),
        ("👥", "Agent Performance", "Agent 3 and Agent 7 consistently outperform - recommend best practice sharing"),
        ("📞", "Call Quality", "Positive sentiment calls convert 34% better - focus on training"),
        ("🌍", "Geographic Focus", "Saudi Arabia market showing recovery trend - increase allocation by 20%"),
        ("📊", "Lead Scoring", "Leads from referrals show 67% higher conversion - prioritize referral program")
    ]
    
    for icon, category, insight in insights:
        st.markdown(f"""
        <div class="alert-success">
            <strong>{icon} {category}:</strong> {insight}
        </div>
        """, unsafe_allow_html=True)
    
    # Model performance metrics
    st.markdown("### 🔧 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prediction Accuracy", "87.3%", delta="2.1%")
        st.metric("Revenue Forecast Error", "±8.5%", delta="-1.2%")
    
    with col2:
        st.metric("Lead Score Precision", "84.7%", delta="1.8%")
        st.metric("Conversion Prediction", "79.2%", delta="3.4%")
    
    with col3:
        st.metric("Model Update Frequency", "Daily", delta="Real-time")
        st.metric("Data Freshness", "< 1 hour", delta="Current")

def show_agent_performance(data, agent_id):
    """Individual agent performance dashboard"""
    st.markdown(f'<div class="section-header"><h2>👤 Agent {agent_id} Performance Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Get agent data
    agent_data = data['agents'][data['agents']['agent_id'] == agent_id].iloc[0]
    agent_calls = data['calls'][data['calls']['agent_id'] == agent_id]
    agent_tasks = data['tasks'][data['tasks']['agent_id'] == agent_id]
    
    # Personal KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Calls Today", agent_data['total_calls_today'], delta="3")
    with col2:
        st.metric("Success Rate", f"{agent_data['success_rate']:.1%}", delta="2.1%")
    with col3:
        st.metric("Revenue Generated", f"SAR {agent_data['revenue_generated']:,.0f}", delta="15.3K")
    with col4:
        st.metric("Customer Satisfaction", f"{agent_data['customer_satisfaction']:.1f}/5.0", delta="0.2")
    
    # Personal performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily call trend (simulated)
        days = pd.date_range(start=datetime.now() - timedelta(days=14), periods=14, freq='D')
        daily_calls = [np.random.randint(5, 20) for _ in days]
        
        fig_trend = px.line(x=days, y=daily_calls, title="📈 My 14-Day Call Trend")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Call outcome distribution
        if len(agent_calls) > 0:
            outcome_counts = agent_calls['outcome'].value_counts()
            fig_outcomes = px.pie(values=outcome_counts.values, names=outcome_counts.index,
                                title="📊 My Call Outcomes")
            st.plotly_chart(fig_outcomes, use_container_width=True)
        else:
            st.info("No call data available for this agent")
    
    # My tasks
    st.markdown("### ✅ My Tasks")
    
    if len(agent_tasks) > 0:
        task_summary = agent_tasks['status'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pending Tasks", task_summary.get('Pending', 0))
        with col2:
            st.metric("In Progress", task_summary.get('In Progress', 0))
        with col3:
            st.metric("Overdue", task_summary.get('Overdue', 0), delta_color="inverse")
        
        st.dataframe(agent_tasks[['task_type', 'priority', 'due_date', 'status', 'client_name']], use_container_width=True)
    else:
        st.info("No tasks assigned to this agent")

def show_team_overview(data):
    """Team lead overview dashboard"""
    st.markdown('<div class="section-header"><h2>👥 Team Overview Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Team KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_team_calls = data['agents']['total_calls_today'].sum()
    avg_team_success = data['agents']['success_rate'].mean()
    total_team_revenue = data['agents']['revenue_generated'].sum()
    available_agents = len(data['agents'][data['agents']['availability'] == 'Available'])
    
    with col1:
        st.metric("Team Calls Today", total_team_calls, delta="12")
    with col2:
        st.metric("Avg Success Rate", f"{avg_team_success:.1%}", delta="1.8%")
    with col3:
        st.metric("Team Revenue", f"SAR {total_team_revenue:,.0f}", delta="25.4K")
    with col4:
        st.metric("Available Agents", f"{available_agents}/10", delta="0")
    with col5:
        st.metric("Team Performance", "87%", delta="3.2%")
    
    # Team performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_team_success = px.bar(data['agents'], x='agent_name', y='success_rate',
                                title="🎯 Team Success Rate Comparison")
        st.plotly_chart(fig_team_success, use_container_width=True)
    
    with col2:
        fig_team_revenue = px.bar(data['agents'], x='agent_name', y='revenue_generated',
                                title="💰 Team Revenue Comparison")
        st.plotly_chart(fig_team_revenue, use_container_width=True)
    
    # Team status
    st.markdown("### 📊 Current Team Status")
    
    team_status = data['agents'][['agent_name', 'availability', 'total_calls_today', 'success_rate', 'customer_satisfaction']].copy()
    team_status['success_rate'] = team_status['success_rate'].apply(lambda x: f"{x:.1%}")
    team_status['customer_satisfaction'] = team_status['customer_satisfaction'].apply(lambda x: f"{x:.1f}/5.0")
    
    st.dataframe(team_status, use_container_width=True)

def show_agent_management(data):
    """Agent management dashboard for team leads"""
    st.markdown('<div class="section-header"><h2>👥 Agent Performance Management</h2></div>', unsafe_allow_html=True)
    
    # Performance rankings
    top_performers = data['agents'].nlargest(3, 'revenue_generated')
    
    st.markdown("### 🏆 Top Performers")
    
    col1, col2, col3 = st.columns(3)
    
    for i, (col, agent) in enumerate(zip([col1, col2, col3], top_performers.itertuples())):
        with col:
            medal = ["🥇", "🥈", "🥉"][i]
            st.markdown(f"""
            <div class="metric-card">
                <h3>{medal} {agent.agent_name}</h3>
                <p><strong>Revenue:</strong> SAR {agent.revenue_generated:,.0f}</p>
                <p><strong>Success Rate:</strong> {agent.success_rate:.1%}</p>
                <p><strong>Satisfaction:</strong> {agent.customer_satisfaction:.1f}/5.0</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Performance matrix
    st.markdown("### 📊 Performance Matrix")
    
    fig_matrix = px.scatter(data['agents'], x='success_rate', y='revenue_generated',
                          size='total_calls_month', hover_name='agent_name',
                          title="📈 Success Rate vs Revenue (Bubble = Total Calls)")
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    # Detailed agent table
    st.markdown("### 📋 Detailed Agent Performance")
    
    detailed_performance = data['agents'].copy()
    detailed_performance['success_rate'] = detailed_performance['success_rate'].apply(lambda x: f"{x:.1%}")
    detailed_performance['revenue_generated'] = detailed_performance['revenue_generated'].apply(lambda x: f"SAR {x:,.0f}")
    detailed_performance['customer_satisfaction'] = detailed_performance['customer_satisfaction'].apply(lambda x: f"{x:.1f}/5.0")
    
    st.dataframe(detailed_performance, use_container_width=True)

def show_overview_dashboard(data, user_role):
    """Default overview dashboard"""
    st.markdown(f'<div class="section-header"><h2>🏠 {user_role} Overview Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Quick summary based on role
    if user_role == "Agent":
        st.markdown("### 👤 My Quick Summary")
        agent_data = data['agents'].iloc[0]  # Assume first agent for demo
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("My Calls Today", agent_data['total_calls_today'])
        with col2:
            st.metric("My Success Rate", f"{agent_data['success_rate']:.1%}")
        with col3:
            st.metric("My Revenue", f"SAR {agent_data['revenue_generated']:,.0f}")
    
    else:
        # Summary for managers/executives
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Leads", len(data['leads']))
        with col2:
            st.metric("Total Calls", data['agents']['total_calls_month'].sum())
        with col3:
            st.metric("Team Success Rate", f"{data['agents']['success_rate'].mean():.1%}")
        with col4:
            st.metric("Total Revenue", f"SAR {data['agents']['revenue_generated'].sum():,.0f}")
    
    # Quick charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Lead status overview
        status_counts = data['leads']['status'].value_counts()
        fig_status = px.pie(values=status_counts.values, names=status_counts.index,
                          title="📊 Lead Status Overview")
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        # Geographic overview
        geo_revenue = data['geographic']['total_revenue']
        fig_geo = px.bar(x=data['geographic']['country'], y=geo_revenue,
                        title="🌍 Revenue by Country")
        st.plotly_chart(fig_geo, use_container_width=True)

if __name__ == "__main__":
    main()