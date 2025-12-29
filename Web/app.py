import streamlit as st
import joblib
import numpy as np

# ===============================
# 0. Page configuration
# ===============================
st.set_page_config(
    page_title="Voriconazole Concentration Estimator",
    layout="wide"
)

st.title("Voriconazole (VCZ) Plasma Concentration Estimator")
st.caption(
    "Estimates voriconazole trough concentration using a machine learning clearance model "
    "combined with an empirical calibration step to match real-world TDM data."
)

# ===============================
# 1. Load models
# ===============================
@st.cache_resource
def load_assets():
    model_cl = joblib.load("Web/xgb_cl.pkl")
    calibrator = joblib.load("Web/calibrator_DV.pkl")
    return model_cl, calibrator

model_cl, calibrator = load_assets()

# ===============================
# 2. Feature order (must match training)
# ===============================
features_cl = ['CRP', 'ALB', 'GenotypingValue', 'Age', 'Sex', 'TBIL', 'Weight']

# ===============================
# 3. Sidebar: All inputs (including target range)
# ===============================
st.sidebar.header("Patient & Treatment Information")

# Dose and timing
daydose = st.sidebar.number_input(
    "Current daily dose (mg/day)",
    min_value=10.0, max_value=1200.0, value=400.0, step=10.0,
    help="Total daily dose (oral or IV)"
)

time_val = st.sidebar.number_input(
    "Days since starting voriconazole",
    min_value=0.0, max_value=60.0, value=7.0, step=0.5
)

st.sidebar.markdown("---")

# Clinical covariates
age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=50)
weight = st.sidebar.number_input("Weight (kg)", min_value=10.0, max_value=200.0, value=70.0, step=0.5)

crp = st.sidebar.number_input("CRP (mg/L)", min_value=0.0, max_value=400.0, value=30.0, step=1.0)
alb = st.sidebar.number_input("Albumin (g/L)", min_value=10.0, max_value=60.0, value=35.0, step=0.5)
tbil = st.sidebar.number_input("Total bilirubin (µmol/L)", min_value=0.0, max_value=500.0, value=12.0, step=1.0)

sex_input = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_input == "Male" else 2

geno_label = st.sidebar.selectbox(
    "CYP2C19 metabolizer status",
    ["NM (Normal metabolizer)", "IM (Intermediate metabolizer)", "PM (Poor metabolizer)"],
    index=0
)
geno_map = {"NM (Normal metabolizer)": 1, "IM (Intermediate metabolizer)": 2, "PM (Poor metabolizer)": 3}
GenotypingValue = geno_map[geno_label]

# Warnings
if crp > 150:
    st.sidebar.warning("High CRP (>150 mg/L): Severe inflammation → reduced clearance")
if alb < 25:
    st.sidebar.info("Low albumin (<25 g/L) → increased voriconazole exposure")

st.sidebar.markdown("---")

# Target concentration range selection (in sidebar → no page flicker)
st.sidebar.subheader("Target Trough Concentration Range")

target_preset = st.sidebar.radio(
    "Clinical target",
    options=[
        "Standard (1.0 - 5.5 mg/L)",
        "Conservative (1.0 - 4.0 mg/L)",
        "Prophylaxis (≥0.5 mg/L)",
        "Custom range"
    ]
)

if target_preset == "Standard (1.0 - 5.5 mg/L)":
    t_low, t_high = 1.0, 5.5
elif target_preset == "Conservative (1.0 - 4.0 mg/L)":
    t_low, t_high = 1.0, 4.0
elif target_preset == "Prophylaxis (≥0.5 mg/L)":
    t_low, t_high = 0.5, 10.0
else:  # Custom
    col1, col2 = st.sidebar.columns(2)
    t_low = col1.number_input("Lower (mg/L)", 0.1, 5.0, 1.0, 0.1, key="custom_low")
    t_high = col2.number_input("Upper (mg/L)", 1.0, 10.0, 5.5, 0.1, key="custom_high")

# ===============================
# 4. PK functions
# ===============================
def theoretical_conc(cl, dose):
    cl_safe = max(cl, 0.1)
    return dose / (24.0 * cl_safe)

def suggest_dose(target_conc, cl):
    cl_safe = max(cl, 0.1)
    return target_conc * 24.0 * cl_safe

# Input array
input_array = np.array([[crp, alb, GenotypingValue, age, sex, tbil, weight]])

# ===============================
# 5. Main prediction
# ===============================
if st.button("Estimate Concentration & Optimize Dose", type="primary", use_container_width=True):

    pred_cl = float(model_cl.predict(input_array)[0])
    theory_conc = theoretical_conc(pred_cl, daydose)
    calibrated_conc = float(calibrator.predict([[theory_conc]])[0])
    calibrated_conc = max(calibrated_conc, 0.1)

    # Main results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted Clearance")
        st.metric("CL/F (L/h)", f"{pred_cl:.3f}")

    with col2:
        st.subheader("Predicted Trough Concentration")
        st.metric("Calibrated concentration (mg/L)", f"{calibrated_conc:.3f}")

        if calibrated_conc < 1.0:
            st.error("⚠️ Subtherapeutic risk (<1 mg/L) – consider increasing dose + urgent TDM")
        elif calibrated_conc > 5.5:
            st.error("⚠️ High neurotoxicity risk (>5.5 mg/L) – consider reducing dose + urgent TDM")
        elif calibrated_conc > 4.0:
            st.warning("Elevated (>4 mg/L) – monitor closely for toxicity")
        else:
            st.success("Within commonly accepted target range")

        st.caption("Final value is empirically calibrated to match observed clinical concentrations (±30% expected variability)")

    st.markdown("---")

    # Dose Optimization
    st.subheader("Dose Optimization Recommendation")

    if t_low >= t_high:
        st.error("Lower target must be less than upper target.")
    else:
        dose_low = suggest_dose(t_low, pred_cl)
        dose_high = suggest_dose(t_high, pred_cl)

        st.markdown(f"""
        **Recommended daily dose to achieve {t_low} – {t_high} mg/L**:
        - **Total dose**: **{dose_low:.0f} – {dose_high:.0f}** mg/day
        - **Weight-based**: **{dose_low/weight:.2f} – {dose_high/weight:.2f}** mg/kg/day
        """)

        if daydose < dose_low:
            st.warning(f"Current dose ({daydose:.0f} mg/day) is **below** recommended range → consider increasing")
        elif daydose > dose_high:
            st.warning(f"Current dose ({daydose:.0f} mg/day) is **above** recommended range → consider reducing")
        else:
            st.success(f"Current dose ({daydose:.0f} mg/day) is **within** recommended range")

    st.markdown("---")

    # Detailed interpretation
    with st.expander("Detailed Model Interpretation"):
        st.write(f"**Theoretical linear concentration** (dose / 24×CL): **{theory_conc:.3f}** mg/L")
        st.write(f"**Calibrated predicted concentration**: **{calibrated_conc:.3f}** mg/L ← **primary value for clinical use**")
        st.write(f"Current dose: **{daydose:.0f}** mg/day (**{daydose/weight:.2f}** mg/kg/day)")
        st.write(f"Days on therapy: **{time_val:.1f}**")

        if time_val < 5:
            st.warning("Very early (<5 days) – not at steady state")
        elif time_val < 7:
            st.info("Approaching steady state")
        else:
            st.success("Likely at steady state (≥7 days)")

# ===============================
# 6. Footer notes
# ===============================
st.markdown("### Important Notes")
st.markdown("""
- This tool is for **research and clinical decision support only**.
- **Always verify with therapeutic drug monitoring (TDM)** when available.
- The final concentration is calibrated against real-world data for improved accuracy.
- Dose recommendations assume steady-state, maintenance dosing, and linear pharmacokinetics.
- Common target: **1.0 – 5.5 mg/L** (some centers prefer ≤4.0 mg/L to reduce toxicity risk).
""")
