import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# Load API Key
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

st.set_page_config(page_title="Custom LLM Analytics App", layout="wide")
st.title("📊 LLM Analytics Based on Your Instruction")

uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded:
    # Load file
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📝 Enter Your Custom Analysis Instruction")

    user_instruction = st.text_area(
        "Example: 'Do EDA', 'Find correlations', 'Build summary', 'Check missing values', 'Generate insights'",
        height=120
    )

    if st.button("Run Custom Analysis"):
        if not user_instruction.strip():
            st.error("Please enter an analysis instruction.")
            st.stop()

        # Prepare dataset sample for LLM
        df_text = df.head(20).to_csv(index=False)

        # Build dynamic prompt
        prompt = f"""
You are a senior data analyst.

User instruction:
\"\"\"{user_instruction}\"\"\"

Perform ONLY what the user asked.
Do not add extra analysis beyond the instruction.

Dataset sample:
{df_text}

Provide results clearly and step-by-step.
"""

        with st.spinner("Running analysis based on your instruction..."):
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

        analysis = response.choices[0].message.content
        st.subheader("📌 Result of Your Custom Analysis")
        st.write(analysis)

        # ------------------------------------------------------------
        # DOWNLOAD TXT FILE
        # ------------------------------------------------------------
        txt_buffer = io.BytesIO()
        txt_buffer.write(analysis.encode())
        txt_buffer.seek(0)

        st.download_button(
            "📥 Download Analysis as TXT",
            data=txt_buffer,
            file_name="Custom_Analysis.txt",
            mime="text/plain"
        )

        # ------------------------------------------------------------
        # DOWNLOAD EXCEL WITH RESULTS
        # ------------------------------------------------------------
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Dataset", index=False)

            # Analysis sheet
            analysis_df = pd.DataFrame({"Analysis": analysis.split("\n")})
            analysis_df.to_excel(writer, sheet_name="LLM_Result", index=False)

        excel_buffer.seek(0)

        st.download_button(
            "📥 Download Analysis as Excel",
            data=excel_buffer,
            file_name="Custom_Analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # ------------------------------------------------------------
    # VISUALIZATION BASED ON INSTRUCTION
    # ------------------------------------------------------------
    if "plot" in user_instruction.lower() or \
       "graph" in user_instruction.lower() or \
       "visual" in user_instruction.lower():

        st.subheader("📈 Visualizations (as requested)")

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        # Histogram
        if "hist" in user_instruction.lower() and num_cols:
            st.markdown("### Histogram Plots")
            for col in num_cols:
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna())
                ax.set_title(f"Histogram: {col}")
                st.pyplot(fig)

        # Correlation Heatmap
        if "correlation" in user_instruction.lower() and len(num_cols) >= 2:
            st.markdown("### 🔥 Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # Bar charts
        if "bar" in user_instruction.lower() and cat_cols:
            st.markdown("### Category Bar Charts")
            for col in cat_cols[:5]:
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind="bar", ax=ax)
                ax.set_title(f"Bar Chart: {col}")
                st.pyplot(fig)
