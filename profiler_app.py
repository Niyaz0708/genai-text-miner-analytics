import streamlit as st
import pandas as pd
import os
import json
import google.generativeai as genai

st.set_page_config(page_title="Advanced AI Data Profiler", layout="wide")

st.title("🤖 GenAI Unstructured Text Miner & Analytics Engine")
st.markdown("*Advanced Enterprise-grade platform developed to extract structured tokens, CSAT scores, and operational metrics from raw text ledgers.*")
st.markdown("***")

# 1. Dynamic Data Uploader
st.subheader("📁 Upload Customer Feedback Dataset")
uploaded_file = st.file_uploader("Choose a CSV file containing customer reviews", type=["csv"])

@st.cache_data
def load_data(file_source):
    return pd.read_csv(file_source)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("✅ Custom CSV uploaded successfully!")
else:
    if os.path.exists("online_retail_feedback.csv"):
        df = load_data("online_retail_feedback.csv")
        st.info("ℹ️ Using default 'online_retail_feedback.csv' benchmark database.")
    else:
        st.error("Please upload a CSV file to test.")
        st.stop()

st.dataframe(df, use_container_width=True)

st.markdown("***")
st.subheader("⚙️ Configure Mining Parameters")
text_column = st.selectbox("Select the column containing Feedback/Reviews text:", df.columns)

st.markdown("***")

# 2. AI Analytics Processing
st.subheader("🔍 Automated AI Text Mining & Visual Insights")
api_key_input = st.text_input("Enter your Gemini API Key to unlock AI Miner:", type="password")

if st.button("🚀 Run AI Deep Analytics Engine"):
    if not api_key_input:
        st.error("Please enter a valid Gemini API Key first!")
    else:
        with st.spinner("AI is executing deep text mining pipelines and structural profiling..."):
            try:
                genai.configure(api_key=api_key_input)
                model = genai.GenerativeModel('gemini-3.5-flash')
                
            
                data_records = df[[text_column]].to_json(orient="records")
                
                
                prompt = f"""
                You are an expert Data Scientist and Text Mining Specialist. Analyze the following customer reviews:
                {data_records}
                
                You must perform 3 analytical tasks:
                1. Provide a professional narrative Executive Summary of customer satisfaction.
                2. Provide Top 2 critical operational complaints.
                3. For EVERY single row index in the data, output a JSON array of objects with keys: "sentiment" (strictly POSITIVE, NEGATIVE, or NEUTRAL), "csat_score" (an integer from 1 to 5), and "category_tag" (strictly one word like Delivery, Price, Quality, Service, or Product).
                
                Format your final answer exactly like this, using the delimiter '|||' to separate text from JSON data:
                [Insert Executive Summary and Top 2 Complaints here in plain text/markdown]
                |||
                [Insert the raw JSON array of objects here]
                """
                
                response = model.generate_content(prompt)
                full_output = response.text
                
                
                text_part, json_part = full_output.split("|||")
                parsed_json = json.loads(json_part.strip().replace("```json", "").replace("```", ""))
                
                
                sentiments = [item["sentiment"] for item in parsed_json]
                csat_scores = [item["csat_score"] for item in parsed_json]
                category_tags = [item["category_tag"] for item in parsed_json]
                
                df['Sentiment'] = sentiments[:len(df)]
                df['CSAT_Score'] = csat_scores[:len(df)]
                df['Topic_Tag'] = category_tags[:len(df)]
                
                st.success("AI Text Mining & Structural Profiling Complete!")
                st.markdown("***")
                
                # 📊 TOP KPI METRICS CARDS
                col_m1, col_m2 = st.columns(2)
                col_m1.metric(label="Average Customer CSAT (Out of 5)", value=f"{df['CSAT_Score'].mean():.2f} / 5.0")
                col_m2.metric(label="Negative Feedback Volume", value=f"{df['Sentiment'].value_counts().get('NEGATIVE', 0)} Rows")
                
                st.markdown("***")
                
               
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("📢 AI Corporate Executive Summary")
                    st.markdown(text_part)
                
                with col_right:
                    st.subheader("📊 Sentiment Analytics Breakdown")
                    st.bar_chart(df['Sentiment'].value_counts())
                    
                    st.subheader("🏷️ Operational Issues by Category Tag")
                    st.bar_chart(df['Topic_Tag'].value_counts())
                
               
                st.markdown("***")
                st.subheader("📥 Export Augmented Metadata")
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Categorized Insights CSV",
                    data=csv_data,
                    file_name="ai_augmented_customer_insights.csv",
                    mime="text/csv"
                )
                    
            except Exception as e:
                st.error(f"Error executing analysis: {str(e)}. Please check your API Key and formatting constraints.")

# Sidebar Portfolio
st.sidebar.markdown("***")
st.sidebar.markdown("""
<div style='text-align: center; color: #888888; font-size: 13px; padding-top: 20px;'>
    <p style='margin-bottom: 2px;'><b>PROJECT TITLE:</b></p>
    <p style='color: #00ffcc; font-size: 14px; margin-top: 0px;'><b>Advanced GenAI Analytics Engine</b></p>
    <p style='margin-bottom: 2px; margin-top: 15px;'><b>DEVELOPER:</b></p>
    <p style='color: #ffffff; font-size: 14px; margin-top: 0px;'>Md Niyaz Alam | Data Analyst With GenAI</p>
</div>
""", unsafe_allow_html=True)
