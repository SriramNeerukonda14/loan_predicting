import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Loan Analyzer", page_icon="📊")

st.title("📊 Loan Data Analyzer & Predictor")

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

# -------------------- MAIN --------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)  # FIXED

    tab1, tab2, tab3 = st.tabs(["📊 Data", "⚙️ Processing", "🤖 Model"])

    # -------------------- TAB 1: DATA --------------------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Basic Info")
        st.write(df.describe())

    # -------------------- TAB 2: PROCESSING --------------------
    with tab2:
        st.subheader("Handling Missing Values")

        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].fillna(df[col].median())

        st.success("Missing values handled")

        st.subheader("Encoding Categorical Data")
        cat_cols = df.select_dtypes(include='object').columns

        if len(cat_cols) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df[cat_cols])

            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(cat_cols)
            )

            df = df.drop(columns=cat_cols)
            df = pd.concat([df, encoded_df], axis=1)

            st.success("Encoding completed")

        st.subheader("Scaling Data")
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df)

        st.success("Scaling applied")
        st.dataframe(df.head())

    # -------------------- TAB 3: MODEL --------------------
    with tab3:
        if "target" in df.columns:
            X = df.drop("target", axis=1)
            y = df["target"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            st.subheader("Train Model")

            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.metric("MSE", round(mse, 3))
            st.metric("R² Score", round(r2, 3))

            # Visualization
            st.subheader("Prediction Graph")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")

            st.pyplot(fig)
        else:
            st.warning("Target column not found")
else:
    st.info("Please upload a dataset to begin")