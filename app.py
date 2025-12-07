import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Disable Arrow (avoids pyarrow requirement)
st._arrow_alpha_columns = False
try:
    st.dataframe.__dict__["_use_arrow"] = False
except:
    pass

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------- PAGE CONFIG --------------------------- #

st.set_page_config(
    page_title="Spam Shield - Email & SMS Classifier",
    page_icon="📧",
    layout="wide"
)

# --------------------------- CUSTOM CSS --------------------------- #

st.markdown(
    """
    <style>
        .stApp { background-color: #f8fafc; }

        .metric-card {
            padding: 20px;
            border-radius: 18px;
            background: white;
            border: 1px solid #e2e8f0;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
            text-align: center;
        }

        .footer-text {
            font-size: 0.78rem;
            color: #64748b;
            text-align: center;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------- LOAD DATA --------------------------- #

@st.cache_data
def load_data(path: str = "mail_data.csv"):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Expect: Category, Message
    if "category" not in df.columns or "message" not in df.columns:
        st.error("❌ CSV must contain 'Category' and 'Message' columns.")
        st.write("Found columns:", list(df.columns))
        st.stop()

    # Rename to standard names used by the model
    df = df.rename(columns={
        "message": "text",
        "category": "label"
    })

    # Clean missing
    df.dropna(subset=["text", "label"], inplace=True)

    # Convert labels ham/spam
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    return df


# --------------------------- TRAIN MODEL --------------------------- #

@st.cache_resource
def train_model(df):
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("nb", MultinomialNB()),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, accuracy, cm


def generate_safe_reply(is_spam, original_text):
    if is_spam:
        return (
            "Hi,\n\n"
            "This message appears to be unsafe or suspicious. I will not share personal "
            "information, passwords or OTPs.\n"
            "If this is related to banking or services, I will contact the official support.\n\n"
            "Regards,\n[Your Name]"
        )
    else:
        return (
            "Hi,\n\n"
            "Thank you for your message. I have received your details and will get back soon.\n\n"
            "Regards,\n[Your Name]"
        )


# --------------------------- LOAD + TRAIN --------------------------- #

with st.spinner("📂 Loading dataset and training model..."):
    df = load_data()
    model, accuracy, cm = train_model(df)

spam_count = int((df["label"] == 1).sum())
ham_count = int((df["label"] == 0).sum())

# --------------------------- SIDEBAR --------------------------- #

st.sidebar.title("📧 Spam Shield")
page = st.sidebar.radio("Navigate", ("🏠 Home", "🔍 Try the Classifier"))

st.sidebar.markdown("---")
st.sidebar.write("Model: **TF-IDF + Multinomial Naive Bayes**")
st.sidebar.write(f"Total emails: **{len(df)}**")

# --------------------------- HOME PAGE --------------------------- #

if page == "🏠 Home":

    # Header
    st.markdown(
        """
        <h1 style="text-align:center; font-size:48px; font-weight:800;">
            🛡️ SpamShield 🔐
        </h1>
        <p style="text-align:center; font-size:18px; color:#555;">
            AI-powered Email & SMS Spam Detection with Safe Reply Generation.
        </p>
        <br>
        """,
        unsafe_allow_html=True
    )

    # --------------------------- FEATURES SECTION --------------------------- #
    
    st.markdown(
        """
    ### 🧩 Features of **SpamShield**
    """)

    # Row 1
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="metric-card" style="text-align:left;">
                <h4>🔍 Smart Spam Detection</h4>
                <p>Uses TF-IDF + Naive Bayes to classify emails & SMS as <b>Spam</b> or <b>Not Spam</b>.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-card" style="text-align:left;">
                <h4>🛡️ Safe Auto-Reply</h4>
                <p>Automatically generates a <b>safe, scam-proof reply</b> when spam is detected.</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-card" style="text-align:left;">
                <h4>📊 Accuracy Display</h4>
                <p>Shows model accuracy <b>and a mini graph</b> right after classification.</p>
            </div><br>
        """, unsafe_allow_html=True)


    # Row 2
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
            <div class="metric-card" style="text-align:left;">
                <h4>⚡ Real-Time Classification</h4>
                <p>Instant detection for any message — Email, SMS, or WhatsApp text.</p>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
            <div class="metric-card" style="text-align:left;">
                <h4>📥 Clean Dataset Handling</h4>
                <p>Automatically processes and maps dataset labels for smooth training.</p>
            </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
            <div class="metric-card" style="text-align:left;">
                <h4>💡 Modern UI</h4>
                <p>Clean, elegant, and user-friendly interface with a premium feel.</p>
            </div><br><br>
        """, unsafe_allow_html=True)


    # ---------------- QUICK STATS ---------------- #
    st.markdown("### 📊 Quick Stats")
    colA, colB, colC, colD = st.columns(4)

    colA.markdown(
        f"""
        <div class="metric-card">
            <h4>Total Messages</h4>
            <p style="font-size:26px; font-weight:700;">{len(df)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    colB.markdown(
        f"""
        <div class="metric-card">
            <h4>Spam Messages</h4>
            <p style="font-size:26px; font-weight:700; color:#dc2626;">{spam_count}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    colC.markdown(
        f"""
        <div class="metric-card">
            <h4>Ham Messages</h4>
            <p style="font-size:26px; font-weight:700; color:#16a34a;">{ham_count}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    colD.markdown(
        f"""
        <div class="metric-card">
            <h4>Accuracy</h4>
            <p style="font-size:26px; font-weight:700; color:#2563eb;">{accuracy*100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # --------------------------- FOOTER --------------------------- #

    st.markdown("""
    <br><br><br>
    <div style="
        text-align:center; 
        padding:20px 0; 
        color:#64748b; 
        font-size:14px;
        border-top:1px solid #e5e7eb;
        margin-top:40px;
    ">
        🚀 Built as a mini-project: <b>SpamShield – Email & SMS Spam Classifier with Safe Reply Drafting.</b><br>
        Made with ❤️ using Python & Streamlit<br>© 2025 SpamShield
    </div>
""", unsafe_allow_html=True)




# --------------------------- TRY THE CLASSIFIER --------------------------- #

elif page == "🔍 Try the Classifier":

    st.markdown("## 🔍 Try the Spam / Scam Classifier")
    st.write("Paste any message or email to classify and auto-generate a safe reply.")

    user_text = st.text_area(
        "Message / Email content",
        height=220,
        placeholder="Example: You won ₹5,00,000 lottery. Send your bank details."
    )

    if st.button("Classify & Draft Reply", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter a message.")
        else:
            pred = model.predict([user_text])[0]
            is_spam = (pred == 1)

            if is_spam:
                st.error("🚨 Spam / Scam Detected")
            else:
                st.success("✅ Not Spam (Safe)")

            st.markdown("### ✉️ Suggested Safe Reply")
            safe_reply = generate_safe_reply(is_spam, user_text)
            st.code(safe_reply)


                # ---------------- Accuracy Graph With Arrow Annotation ----------------
        st.markdown("### 📊 Model Accuracy")
        st.success(f"Model Test Accuracy: **{accuracy * 100:.2f}%**")

        fig, ax = plt.subplots(figsize=(2.4, 1.8))  # small graph

        acc_value = accuracy * 100

        # Bar plot
        ax.bar(["Accuracy"], [acc_value], color="#2563eb")

        # Y-axis formatting
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)", fontsize=8)

        # Clean X-axis
        
        ax.set_xlabel(
    "Accuracy Graph ",
    fontsize=15,
    color="#2563eb",
    fontweight="bold",
    labelpad=12 
        )      

        # Arrow annotation to the right
        ax.annotate(
            f"{acc_value:.2f}%",             # text shown
            xy=(0, acc_value),               # point on bar
            xytext=(0.1, acc_value + 5),     # text position slightly right & above
            textcoords='data',
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=8
        )

        # Clean look
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Center the graph in page
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.pyplot(fig)
