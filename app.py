import streamlit as st
import joblib
import re
from PIL import Image
import pytesseract

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_model():
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

st.markdown("""
<style>
.stApp {
    background-color: #0f1117;
    color: white;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}
.card {
    background-color: #172033;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #2d3748;
}
.result-box {
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    font-size: 28px;
    font-weight: 800;
    margin-top: 25px;
}
.fake {
    background-color: #fee2e2;
    color: #991b1b;
}
.real {
    background-color: #dcfce7;
    color: #166534;
}
.warning {
    background-color: #fef3c7;
    color: #92400e;
}
.stTextArea textarea {
    background-color: #1f2430;
    color: white;
    border-radius: 12px;
}
.stButton button {
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📰 Fake News Detector for Students</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Check whether a news article is Real, Fake, or just a short normal statement.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    input_method = st.radio(
        "Choose Input Method",
        ["Type / Paste News", "Upload News Image"],
        horizontal=True
    )

    extracted_text = ""

    if input_method == "Type / Paste News":
        news = st.text_area(
            "Enter News Article",
            height=250,
            placeholder="Paste or type news article here..."
        )

    else:
        uploaded_image = st.file_uploader(
            "Upload photo of news article",
            type=["png", "jpg", "jpeg"]
        )

        news = ""

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded News Image", use_container_width=True)

            if st.button("📸 Extract Text from Image", use_container_width=True):
                extracted_text = pytesseract.image_to_string(image)
                st.session_state["extracted_text"] = extracted_text

        if "extracted_text" in st.session_state:
            news = st.text_area(
                "Extracted Text",
                value=st.session_state["extracted_text"],
                height=250
            )

    check_btn = st.button("🔍 Check News", use_container_width=True)

with col2:
    st.markdown("""
    <div class="card">
    <h4>ℹ️ How it works</h4>
    <p>This AI tool helps students identify fake or real news using Machine Learning and NLP.</p>
    <br>
    <b>Steps:</b>
    <ol>
        <li>Paste news or upload image</li>
        <li>Extract text if image is uploaded</li>
        <li>Click Check News</li>
        <li>View result</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

if check_btn:
    if news.strip() == "":
        st.warning("Please enter news text or upload a news image first.")

    elif len(news.split()) < 5:
        st.markdown(
            '<div class="result-box warning">⚠️ Too Short Text<br>This is not enough content for AI prediction.</div>',
            unsafe_allow_html=True
        )

    elif len(news.split()) < 12:
        st.markdown(
            '<div class="result-box real">ℹ️ Not Fake News<br>This looks like a short normal statement, not a full news article.</div>',
            unsafe_allow_html=True
        )

    else:
        cleaned = clean_text(news)
        vectorized_text = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized_text)[0]

        if hasattr(model, "decision_function"):
            score = model.decision_function(vectorized_text)[0]
            confidence = min(abs(score) * 20, 100)
        else:
            confidence = 80

        if prediction == "FAKE":
            st.markdown(
                f'<div class="result-box fake">🚨 Fake News Detected<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box real">✅ Real News Detected<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )

    st.subheader("📝 Short Summary")
    words = news.split()
    if len(words) > 0:
        summary = " ".join(words[:60])
        st.write(summary + "..." if len(words) > 60 else summary)