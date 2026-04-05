import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Learning Preference Survey",
    page_icon="🎓",
    layout="wide"
)

# Apply Seaborn theme for the plots
sns.set_theme(style="whitegrid")

# ==========================================
# 2. DATA PERSISTENCE LAYER
# ==========================================
# File path for the local database
DATA_FILE = "survey_results.csv"


def load_existing_data():
    """Loads data from CSV or creates a new DataFrame if file doesn't exist."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        # Define the schema for the data
        return pd.DataFrame(columns=[
            'Age', 'Preferred_Mode', 'Avg_Daily_Study_Hours',
            'Engagement_Level', 'Internet_Issue', 'Understanding_Rating'
        ])


def save_new_response(data_dict):
    """Appends a new user response to the CSV file."""
    df = load_existing_data()
    new_row = pd.DataFrame([data_dict])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    return df


# ==========================================
# 3. USER INTERFACE (SIDEBAR)
# ==========================================
st.sidebar.title("📝 Student Survey")
st.sidebar.markdown("Please provide your feedback on learning modes.")

with st.sidebar.form("survey_form", clear_on_submit=True):
    user_name = st.text_input("Full Name")

    user_email = st.text_input("Email Address")

    age = st.number_input("What is your age?", min_value=10, max_value=100, value=20)

    mode = st.selectbox(
        "Preferred Learning Mode",
        options=["Online", "Offline", "Hybrid"]
    )

    hours = st.slider("Average daily study hours", 0.0, 15.0, 4.0, step=0.5)

    engagement = st.select_slider(
        "Engagement Level (1 = Low, 10 = High)",
        options=list(range(1, 11)),
        value=5
    )

    internet = st.radio("Do you face frequent internet issues?", ["Yes", "No"])

    understanding = st.slider("Rate your understanding of topics (1-10)", 1, 10, 5)

    submit_button = st.form_submit_button("Submit Response")

# ==========================================
# 4. MAIN DASHBOARD LOGIC
# ==========================================
st.title("🎓 Online vs. Offline Learning Analysis")
st.markdown("""
This dashboard analyzes real-time student feedback to compare the effectiveness 
of different learning environments. Submit your data in the sidebar to update the charts!
""")

# Handle Submission
if submit_button:
    if user_name and user_email:

        current_response = {
            'Name': user_name,  # New field
            'Email': user_email,  # New field
            'Age': age,
            'Preferred_Mode': mode,
            'Avg_Daily_Study_Hours': hours,
            'Engagement_Level': engagement,
            'Internet_Issue': internet,
            'Understanding_Rating': understanding
        }
        df = save_new_response(current_response)
        st.success("Thank you! Your response has been recorded.")
    else:
        st.error("Please provide both your name and email before submitting.")
        df = load_existing_data()

    # ==========================================
    # 5. DATA VISUALIZATION
    # ==========================================
    if not df.empty:
        # --- Metric Row ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Responses", len(df))
        col2.metric("Avg Engagement", f"{df['Engagement_Level'].mean():.1f}/10")
        col3.metric("Avg Study Hours", f"{df['Avg_Daily_Study_Hours'].mean():.1f} hrs")
    
        st.divider()
    
        # --- Charts Row 1 ---
        chart_col1, chart_col2 = st.columns(2)
    
        with chart_col1:
            st.subheader("Distribution of Preferences")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, x='Preferred_Mode', palette='viridis', ax=ax1)
            ax1.set_ylabel("Number of Students")
            st.pyplot(fig1)
    
        with chart_col2:
            st.subheader("Understanding Rating by Mode")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x='Preferred_Mode', y='Understanding_Rating', palette='Set2', ax=ax2)
            st.pyplot(fig2)
    
        # --- Charts Row 2 ---
        st.subheader("Impact of Internet Issues on Engagement")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=df, x='Preferred_Mode', y='Engagement_Level', hue='Internet_Issue', ax=ax3)
        st.pyplot(fig3)
    
        # --- Raw Data ---
        with st.expander("View Raw Data Table"):
            st.dataframe(df, use_container_width=True)
    
    else:
        st.info("👋 Welcome! No data has been collected yet. Use the sidebar to submit the first response.")

# ==========================================
# 6. DOCUMENTATION & FOOTER
# ==========================================
st.sidebar.divider()
st.sidebar.info("""
**How to use:**
1. Fill out the form.
2. Click **Submit**.
3. Watch the dashboard update!

**Tech Stack:**
- Streamlit (Frontend)
- Pandas (Data Processing)
- Seaborn/Matplotlib (Analytics)
""")
