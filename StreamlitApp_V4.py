import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="ChurnShield AI",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- DATA LOADING ----------
DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ENSEMBLE_FILE = "ensemble_pipeline.joblib"
MODELS = {
    "CatBoost": "cat_model.joblib",
    "XGBoost": "xgb_model.joblib",
    "LightGBM": "lgbm_model.joblib",
}

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, index_col="customerID")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)
    return df

@st.cache_resource
def load_ensemble():
    try:
        return joblib.load(ENSEMBLE_FILE)
    except:
        return None

@st.cache_resource
def load_individual_models():
    models = {}
    for name, path in MODELS.items():
        try:
            models[name] = joblib.load(path)
        except:
            models[name] = None
    return models

df = load_data()
ensemble = load_ensemble()
individual_models = load_individual_models()

# ---------- HELPER: Extract importances ----------
def extract_importance(estimator, feature_names):
    if hasattr(estimator, 'steps'):
        estimator = estimator.steps[-1][1]
    try:
        if hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            raw = np.abs(coef[0]) if coef.shape[0] == 1 else np.abs(coef).mean(axis=0)
        elif hasattr(estimator, 'feature_importances_'):
            raw = estimator.feature_importances_
        elif hasattr(estimator, 'get_feature_importance'):
            raw = estimator.get_feature_importance()
        else:
            return None
        return pd.DataFrame({'feature': feature_names, 'importance': raw}).sort_values('importance', ascending=True)
    except:
        return None

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("📉 ChurnShield AI")
    st.divider()
    page = st.radio("Navigation", ["📝 Blog & Findings", "📊 Analysis", "🧪 Test Model"], index=0)
    st.divider()
    st.caption("End‑to‑end ML project\nby Shodolamu Opeyemi")

# ================== PAGE: BLOG & FINDINGS ==================
# ================== PAGE: BLOG & FINDINGS ==================
if page == "📝 Blog & Findings":
    st.header("🚀 ChurnShield AI: Predicting Customer Churn Before It Happens")
    st.subheader("An End‑to‑End Machine Learning Journey", divider="grey")

    # ---------- 1. WHAT IS CUSTOMER CHURN? ----------
    with st.container(border=True):
        st.subheader("📖 What Is Customer Churn?")
        st.markdown("""
        **Customer churn** — also known as customer attrition — is the rate at which customers stop doing business with a company over a given period of time.

        For a telecom provider, churn happens when a subscriber cancels their service and moves to a competitor, or simply disconnects without renewing. 
        While "churn" might sound like a simple metric, its impact is anything but small.

        **Why does churn matter?**
        - Every lost customer represents **lost recurring revenue**.
        - Acquiring a new customer costs **5 to 25 times more** than retaining an existing one (source: Harvard Business Review).
        - Even a small increase in churn can wipe out millions in annual revenue for mid‑sized companies.
        - High churn signals deeper problems: poor service quality, pricing misalignment, or unmet customer needs.

        In a typical telecom company with 1 million subscribers and a monthly churn rate of just 2%, **20,000 customers leave every single month**. 
        Over a year, that's nearly a quarter of the entire customer base — gone.
        """)

    # ---------- 2. THE PROBLEM ----------
    with st.container(border=True):
        st.subheader("🎯 The Problem I Set Out to Solve")
        st.markdown("""
        Most companies operate **reactively** when it comes to churn:
        - They notice revenue declining.
        - They run reports on who already left.
        - They try to win back lost customers — a costly, uphill battle.

        This reactive approach is inefficient. By the time a customer has cancelled, it's often too late.

        **The core question I wanted to answer:**
        > *"Can we identify customers who are about to churn — before they actually leave?"*

        If the answer is yes, the business can intervene **proactively**:
        - Offer a personalised discount.
        - Upgrade their service or fix a pain point.
        - Simply reach out and re‑engage the customer.

        The challenge is that churn signals are often subtle and spread across many data points: 
        contract type, monthly spending, tenure, service usage, payment method, and more. 
        A human analyst can't spot all these patterns at scale — but a well‑trained machine learning model can.

        **The goal of this project:** Build an intelligent system that, given a customer's profile, predicts their likelihood of churning 
        and explains *why* — empowering retention teams to act with precision.
        """)

    # ---------- 3. THE BENEFIT ----------
    with st.container(border=True):
        st.subheader("💎 The Benefits of Solving This Problem")
        st.markdown("""
        A reliable churn prediction system delivers value across the entire organisation:

        **1. Revenue Protection**
        - Retaining just 5% more customers can increase profits by 25–95% (source: Bain & Company).
        - For a telecom generating $50M annually, a 1% reduction in churn could mean **$500,000 in saved revenue per year**.

        **2. Smarter Marketing Spend**
        - Instead of blanketing all customers with generic retention offers, marketing teams can target only the high‑risk segment.
        - This reduces campaign costs and avoids annoying loyal customers with unnecessary "please stay" discounts.

        **3. Improved Customer Experience**
        - Proactive outreach — like fixing a technical issue before the customer complains — builds loyalty and trust.
        - Happy customers stay longer, spend more, and refer others.

        **4. Data‑Driven Decision Making**
        - Understanding *why* customers churn (e.g., "month‑to‑month contracts", "no tech support") gives product and pricing teams actionable insights.
        - The business can fix root causes instead of just treating symptoms.

        **5. Competitive Advantage**
        - In saturated markets like telecom, retaining existing customers is often the only sustainable growth strategy.
        - Companies that predict and prevent churn outperform competitors who simply react to it.

        In short: **predicting churn isn't just a modelling exercise — it's a direct lever for profitability and growth.**
        """)

    # ---------- 4. WHY THIS MATTERS (REAL‑WORLD CONTEXT) ----------
    with st.container(border=True):
        st.subheader("💡 Why This Matters (Real‑World Context)")
        st.markdown("""
        Customer churn silently drains millions from subscription businesses. 
        For a telecom with 1 million customers and a 2% monthly churn rate, that's 20,000 lost accounts every month.
        Acquiring a new customer costs 5‑25× more than keeping an existing one.

        **Predicting churn before it happens** turns a reactive cost center into a proactive retention engine.
        This project demonstrates that transformation — from raw data to a production‑ready prediction tool.
        """)

    # ---------- 5. KEY FINDINGS & LESSONS LEARNED ----------
    st.subheader("🔬 Key Findings & Lessons Learned", divider="grey")

    with st.container(border=True):
        st.subheader("🔍 The Metric Trap")
        st.markdown("""
        I initially celebrated 80% accuracy — but with 73% non‑churners, a model that always says "No Churn" gets 73%.
        **Accuracy blinded me.** The real business question is: *How many actual leavers do we catch?*
        That's **recall**, and **F1** balances recall with precision.
        """)
        st.success("✅ **Takeaway:** Align your metric with the cost of being wrong. In churn, missing a leaver is far costlier than a false alarm.")

    with st.container(border=True):
        st.subheader("⚖️ Fighting Imbalance with SMOTE")
        st.markdown("""
        Random oversampling duplicated minority examples → overfitting. Random undersampling discarded valuable data → worse.
        I used **SMOTE** (Synthetic Minority Over‑sampling Technique), which synthesises new churn examples by interpolating between real ones.
        Crucially, SMOTE lives **inside the pipeline**, so leakage is impossible — it's applied only to training folds during cross‑validation.
        """)

    with st.container(border=True):
        st.subheader("🚫 The Silent Danger of Pre‑Encoding")
        st.markdown("""
        Applying `OrdinalEncoder` to the whole dataset before splitting lets the encoder "see" the test set's category distribution.
        That leakage artificially inflates performance. Now encoding is locked inside the pipeline, fitted only on training folds.
        This is a subtle but critical lesson every data scientist must internalise.
        """)

    with st.container(border=True):
        st.subheader("🧠 Ensemble Power")
        st.markdown("""
        After tuning CatBoost, XGBoost, and LightGBM individually (all optimised for F1 with SMOTE), I combined them into a **soft‑voting ensemble**.
        The ensemble averages predicted probabilities, yielding more stable, robust predictions than any single model — now production‑ready.
        """)

    # ---------- 6. PROJECT STRUCTURE ----------
    with st.container(border=True):
        st.subheader("🗺️ How This App Is Organised")
        st.markdown("""
        - **📝 Blog & Findings** *(this page)* – The full project narrative: what churn is, the problem, the benefits, and lessons learned.
        - **📊 Analysis** – Interactive data exploration: visualisations of churn drivers, feature importances, and ensemble performance metrics.
        - **🧪 Test Model** – A live prediction tool: enter a customer's profile and get an instant churn risk score with SHAP explanations.
        """)

    st.divider()
    st.caption("This blog is a living document — part of the complete project portfolio. All code and detailed steps are available on GitHub.")
# ================== PAGE: ANALYSIS ==================
elif page == "📊 Analysis":
    st.header("📊 Data Story & Model Insights")

    # ---- 1. Class Imbalance ----
    with st.container(border=True):
        st.subheader("1. Class Imbalance")
        st.caption("The target variable is heavily skewed. Understanding this imbalance is essential because accuracy can be misleading.")
        churn_counts = df["Churn"].value_counts(normalize=True).reset_index()
        churn_counts.columns = ["Churn", "Proportion"]
        fig1 = px.bar(churn_counts, x="Churn", y="Proportion", color="Churn",
                      text=churn_counts["Proportion"].apply(lambda x: f'{x:.1%}'),
                      color_discrete_map={"No": "#3b82f6", "Yes": "#ef4444"})
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        st.info("🔍 **Key insight:** Only 27% churned. We must use SMOTE and focus on recall/F1, not accuracy.")

    # ---- 2. Contract vs TotalCharges ----
    with st.container(border=True):
        st.subheader("2. Contract Type vs. Total Charges (by Churn)")
        st.caption("Contract type strongly influences lifetime value. This boxplot shows total charges split by contract and churn.")
        fig2 = px.box(df, x="Contract", y="TotalCharges", color="Churn",
                      color_discrete_map={"No": "#3b82f6", "Yes": "#ef4444"})
        st.plotly_chart(fig2, use_container_width=True)
        st.info("🔍 **Key insight:** Month‑to‑month customers churn far more, despite lower total charges. Long‑term contracts stick.")

    # ---- 3. Monthly Charges by Churn & Gender ----
    with st.container(border=True):
        st.subheader("3. Monthly Charges Distribution by Churn & Gender")
        st.caption("Do men and women churn at different price points?")
        fig3 = px.violin(df, y="MonthlyCharges", x="Churn", color="gender", box=True, points=False,
                         color_discrete_map={"Male": "#3b82f6", "Female": "#f59e0b"})
        st.plotly_chart(fig3, use_container_width=True)
        st.info("🔍 **Key insight:** Both genders behave similarly; churners pay more. Price sensitivity is gender‑independent.")

    # ---- 4. Tenure vs Monthly Charges Density ----
    with st.container(border=True):
        st.subheader("4. Tenure vs. Monthly Charges Density")
        st.caption("Where do churners cluster in the tenure–charge space?")
        fig4 = px.density_contour(df, x="tenure", y="MonthlyCharges", color="Churn",
                                  marginal_x="histogram", marginal_y="histogram",
                                  color_discrete_map={"No": "#3b82f6", "Yes": "#ef4444"})
        st.plotly_chart(fig4, use_container_width=True)
        st.info("🔍 **Key insight:** The danger zone is low tenure + high monthly charges. Long‑tenure customers rarely churn.")

    # ---- 5. Feature Importances ----
    with st.container(border=True):
        st.subheader("5. Feature Importances (After Tuning)")
        st.caption("What drives churn according to each model in our ensemble?")

        feature_names = df.drop("Churn", axis=1).columns.tolist()
        importances_found = False

        if ensemble is not None and hasattr(ensemble, 'named_steps'):
            voting = ensemble.named_steps.get('voting')
            if voting is not None:
                for name, clf in voting.named_estimators_.items():
                    imp_df = extract_importance(clf, feature_names)
                    if imp_df is not None:
                        importances_found = True
                        st.markdown(f"**{name}**")
                        fig = px.bar(imp_df.tail(10), x='importance', y='feature', orientation='h',
                                     color_discrete_sequence=['#3b82f6'])
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

        if not importances_found:
            for name, model in individual_models.items():
                if model is not None:
                    imp_df = extract_importance(model, feature_names)
                    if imp_df is not None:
                        importances_found = True
                        st.markdown(f"**{name}**")
                        fig = px.bar(imp_df.tail(10), x='importance', y='feature', orientation='h',
                                     color_discrete_sequence=['#3b82f6'])
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

        if not importances_found:
            st.info("No models loaded – feature importance unavailable.")
        else:
            st.info("🔍 **Key insight:** All three models agree: **tenure** and **contract type** dominate. Monthly charges and internet service are secondary.")

    # ---------- MODEL PERFORMANCE ----------
    with st.container(border=True):
        st.subheader("📈 Ensemble Model Performance")
        st.caption("Soft‑voting ensemble (CatBoost + XGBoost + LightGBM) evaluated on a held‑out test set (20% of data).")

        if ensemble is not None:
            target = "Churn"
            mapping = {'No': 0, 'Yes': 1}
            X_eval = df.drop(columns=target)
            y_eval = df[target].map(mapping)
            X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
                X_eval, y_eval, test_size=0.2, stratify=y_eval, random_state=42
            )
            ensemble.fit(X_train_eval, y_train_eval)
            y_pred = ensemble.predict(X_test_eval)

            cm = confusion_matrix(y_test_eval, y_pred)
            report = classification_report(y_test_eval, y_pred, target_names=['No Churn', 'Churn'], output_dict=True)
            acc = report['accuracy']
            rec = report['Churn']['recall']
            prec = report['Churn']['precision']
            f1 = report['macro avg']['f1-score']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.1%}")
            col2.metric("Recall (Churn)", f"{rec:.1%}")
            col3.metric("Precision (Churn)", f"{prec:.1%}")
            col4.metric("F1‑Score", f"{f1:.1%}")

            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               x=['Predicted No', 'Predicted Yes'],
                               y=['Actual No', 'Actual Yes'],
                               title="Confusion Matrix")
            fig_cm.update_layout(width=450, height=400)
            st.plotly_chart(fig_cm, use_container_width=False)

            st.markdown("**Cross‑validation (5‑fold):** F1 = `0.6373 ± 0.0173`, Recall = `0.7411 ± 0.0219`")

            st.success(f"""
            📊 **What these numbers mean for the business:**
            - **Recall {rec:.0%}** – we catch {rec*100:.0f}% of actual leavers, giving us time to intervene.
            - **Precision {prec:.0%}** – when we flag someone as high risk, we're right {prec*100:.0f}% of the time, minimising nuisance.
            - **F1 {f1:.0%}** – a healthy balance between catching churners and not annoying loyal customers.

            In a real campaign, this ensemble could **save thousands of customers per year** by enabling targeted retention at the right moment.
            """)
        else:
            st.warning("Ensemble model not loaded. Ensure 'ensemble_pipeline.joblib' exists.")

# ================== PAGE: TEST MODEL ==================
elif page == "🧪 Test Model":
    st.header("🧪 Live Churn Prediction")

    if ensemble is None:
        st.error("Ensemble model not found. Please train and save it as 'ensemble_pipeline.joblib'.")
        st.stop()

    pipeline = ensemble
    feature_cols = df.drop("Churn", axis=1).columns.tolist()

    with st.container(border=True):
        st.subheader("📝 Customer Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        with col2:
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        with col3:
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure)*MonthlyCharges)

    input_data = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
        "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
        "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])[feature_cols]

    if st.button("🔮 Predict Churn", type="primary"):
        with st.spinner("Analysing customer..."):
            proba = pipeline.predict_proba(input_data)[0]
            pred = pipeline.predict(input_data)[0]
            churn_prob = proba[1] if pipeline.classes_[1] == 1 else proba[0]

            with st.container(border=True):
                if pred == 1:
                    st.error(f"⚠️ Churn Risk: {churn_prob:.1%}")
                else:
                    st.success(f"✅ Low Risk: {churn_prob:.1%}")

            # SHAP explanation
            try:
                encoder = pipeline.named_steps['encoder']
                scaler = pipeline.named_steps['scaler']
                voting = pipeline.named_steps['voting']

                X_enc = encoder.transform(input_data)
                X_scaled = scaler.transform(X_enc)

                shap_vals = []
                for clf in voting.named_estimators_.values():
                    explainer = shap.TreeExplainer(clf)
                    sv = explainer.shap_values(X_scaled)
                    if isinstance(sv, list):
                        sv = sv[1]
                    shap_vals.append(sv)
                avg_shap = np.mean(shap_vals, axis=0)
                shap_df = pd.DataFrame(avg_shap, columns=feature_cols)
                contrib = shap_df.iloc[0].sort_values(ascending=False)

                st.subheader("🔍 Key Drivers")
                col_pos, col_neg = st.columns(2)
                with col_pos:
                    st.markdown("**Pushing towards CHURN**")
                    pos = contrib.head(5)
                    fig_pos = px.bar(x=pos.values, y=pos.index, orientation='h',
                                     color_discrete_sequence=['#ef4444'])
                    fig_pos.update_layout(margin=dict(t=0, b=0), height=200)
                    st.plotly_chart(fig_pos, use_container_width=True)
                with col_neg:
                    st.markdown("**Protecting against churn**")
                    neg = contrib.tail(5).iloc[::-1]
                    fig_neg = px.bar(x=neg.values, y=neg.index, orientation='h',
                                     color_discrete_sequence=['#10b981'])
                    fig_neg.update_layout(margin=dict(t=0, b=0), height=200)
                    st.plotly_chart(fig_neg, use_container_width=True)
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

# ---------- FOOTER ----------
st.divider()
st.caption("© 2026 ChurnShield AI – Built with ❤️ using Streamlit, CatBoost, XGBoost, LightGBM, and SHAP.")
