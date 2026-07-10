import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

from keras.models import load_model
model_path = "cnn_model1.h5"
model = load_model(model_path, compile=False)

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️", layout="wide")

# ----------------------------------------------------------------------------
# One UI dark theme design tokens
#   bg        #0C0C0E  true-black-ish surface, One UI's dark mode base
#   card      #1C1C1F  raised elevated surface
#   primary   #4C8DFF  brightened interaction blue for dark contrast
#   primary_d #6EA2FF  hover / lighter variant
#   ink       #F2F3F5  primary text 
#   ink_soft  #9AA1AC  secondary text
#   line      #2C2D30  hairline dividers
# ----------------------------------------------------------------------------

ONE_UI_DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600;700&family=Caveat:wght@400;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: #0C0C0E;
}

/* kill default streamlit padding so the card system controls spacing */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 100%;
    width: 100%;
}

/* ---------- header ---------- */
.oneui-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 28px;
}
.oneui-header .icon {
    width: 56px;
    height: 56px;
    border-radius: 18px;
    background: linear-gradient(135deg, #4C8DFF 0%, #7AB0FF 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    box-shadow: 0 8px 22px rgba(76, 141, 255, 0.35);
    flex-shrink: 0;
}
.oneui-header h1 {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 30px;
    color: #F2F3F5;
    margin: 0;
    line-height: 1.2;
}
.oneui-header p {
    font-size: 15px;
    color: #9AA1AC;
    margin: 2px 0 0 0;
}

/* ---------- card shell ---------- */
.oneui-card {
    background: #1C1C1F;
    border: 1px solid #2C2D30;
    border-radius: 28px;
    padding: 24px 24px 20px 24px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    margin-bottom: 20px;
}
.oneui-card h3 {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 17px;
    color: #F2F3F5;
    margin: 0 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.oneui-eyebrow {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #4C8DFF;
    text-transform: uppercase;
    margin-bottom: 4px;
    display: block;
}

/* ---------- canvas wrapper: rounds the drawing surface ---------- */
[data-testid="stVerticalBlockBorderWrapper"] canvas {
    border-radius: 20px !important;
}
.canvas-frame {
    border-radius: 22px;
    overflow: hidden;
    border: 1px solid #2C2D30;
    width: fit-content;
    margin: 0 auto 16px auto;
}

/* ---------- buttons, One UI pill style ---------- */
.stButton > button {
    width: 100%;
    border-radius: 999px;
    background: #4C8DFF;
    align-items: center;
    color: #0C0C0E;
    font-weight: 1000;
    font-size: 15px;
    padding: 0.65em 1em;
    border: none;
    box-shadow: 0 6px 18px rgba(76, 141, 255, 0.35);
    transition: transform 0.08s ease, background 0.15s ease;
}
.stButton > button:hover {
    background: #6EA2FF;
    transform: translateY(-1px);
    color: #0C0C0E;
}
.stButton > button:active {
    transform: translateY(0px) scale(0.98);
}

/* ---------- empty state ---------- */
.oneui-empty {
    text-align: center;
    padding: 40px 12px;
    color: #6B7280;
    font-size: 14px;
}
.oneui-empty .glyph {
    font-size: 34px;
    margin-bottom: 8px;
}

/* ---------- big result badge ---------- */
.oneui-result {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 18px;
}
.oneui-digit-badge {
    width: 84px;
    height: 84px;
    border-radius: 24px;
    background: linear-gradient(135deg, #4C8DFF 0%, #7AB0FF 100%);
    color: #0C0C0E;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 10px 26px rgba(76, 141, 255, 0.4);
    flex-shrink: 0;
}
.oneui-result-text .label {
    font-size: 13px;
    color: #9AA1AC;
    font-weight: 500;
    margin-bottom: 2px;
}
.oneui-result-text .conf {
    font-size: 20px;
    font-weight: 700;
    color: #F2F3F5;
    font-family: 'Poppins', sans-serif;
}

/* ---------- confidence bar ---------- */
.oneui-progress-track {
    width: 100%;
    height: 10px;
    background: #2C2D30;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 10px;
}
.oneui-progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #4C8DFF, #7AB0FF);
}

/* ---------- small preview thumbnail row ---------- */
.oneui-thumb-caption {
    font-size: 12px;
    color: #6B7280;
    text-align: center;
    margin-top: 6px;
}

hr {
    border-color: #2C2D30;
}

/* ---------- handwritten notes style ---------- */
.handwritten-notes {
    background: #1C1C1F;
    border: 2px dashed #4C8DFF;
    border-radius: 12px;
    padding: 20px 18px;
    margin-top: 0;
    box-shadow: 0 4px 12px rgba(76, 141, 255, 0.2), inset 0 1px 0px rgba(76, 141, 255, 0.1);
    font-family: 'Caveat', 'Segoe Print', cursive;
    position: relative;
}

.handwritten-notes::before {
    content: '';
    position: absolute;
    top: -8px;
    left: 20%;
    width: 30px;
    height: 30px;
    background: #1C1C1F;
    border-radius: 50%;
    box-shadow: 70px 0 0 -5px #1C1C1F, 140px 0 0 -5px #1C1C1F;
    z-index: 1;
}

.handwritten-notes h3 {
    font-family: 'Caveat', 'Segoe Print', cursive;
    font-size: 28px;
    color: #4C8DFF;
    margin: 0 0 14px 0;
    font-weight: 700;
    transform: rotate(-2deg);
}

.handwritten-notes p {
    font-family: 'Caveat', 'Segoe Print', cursive;
    font-size: 16px;
    color: #9AA1AC;
    margin: 10px 0;
    line-height: 1.8;
    transform: rotate(-1deg);
}

.handwritten-notes strong {
    color: #7AB0FF;
    font-weight: 700;
}

.handwritten-notes .note-item {
    margin: 12px 0;
    padding-left: 16px;
    border-left: 3px solid #4C8DFF;
}

.handwritten-notes .note-emoji {
    font-size: 20px;
    margin-right: 6px;
}
</style>
"""

st.markdown(ONE_UI_DARK_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="oneui-header">
        <div class="icon">✍️</div>
        <div>
            <h1>Digit Recognizer</h1>
            <p>Draw a digit from 0–9 and let the model take a guess.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns([4.0, 3.5, 3.5], gap="large")

with cols[0]:
    st.markdown(
        """
        <div class="handwritten-notes" style="height: 100%; display: flex; flex-direction: column; justify-content: flex-start;">
            <h3>📝 How to Use</h3>
            <div class="note-item">
                <span class="note-emoji">1️⃣</span>
                <strong>Draw a digit</strong><br>
                <span style="font-size: 15px;">Use your mouse to draw a digit from 0–9. Write it clearly!</span>
            </div>
            <div class="note-item">
                <span class="note-emoji">2️⃣</span>
                <strong>Click Predict</strong><br>
                <span style="font-size: 15px;">Hit the blue Predict button to let the AI guess.</span>
            </div>
            <div class="note-item">
                <span class="note-emoji">3️⃣</span>
                <strong>Check Results</strong><br>
                <span style="font-size: 15px;">See the predicted digit & confidence score!</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cols[1]:
    st.markdown(
        """
        <div class="oneui-card">
            <span class="oneui-eyebrow">Step 1</span>
            <h3>🖊️ Draw</h3>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="canvas-frame">', unsafe_allow_html=True)
    canvas = st_canvas(
        fill_color="#45474C",
        stroke_width=10,
        stroke_color="#F2F3F5",
        background_color="#45474C",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("Predict")
    st.markdown("</div>", unsafe_allow_html=True)

with cols[2]:
    st.markdown(
        """
        <div class="oneui-card">
            <span class="oneui-eyebrow">Step 2</span>
            <h3>🔮 Result</h3>
        """,
        unsafe_allow_html=True,
    )

    result_placeholder = st.empty()
    img_placeholder = st.empty()
    plot_placeholder = st.empty()

    if not predict_btn:
        result_placeholder.markdown(
            """
            <div class="oneui-empty">
                <div class="glyph">🎨</div>
                Draw a digit on the left, then tap Predict.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

if predict_btn:
    if canvas and canvas.image_data is not None:
        img = canvas.image_data[:, :, :3]

        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (28, 28))
        gray = gray.astype("float32") / 255.0

        # display the preprocessed grayscale (enlarged for visibility)
        display_img = (gray * 255).astype(np.uint8)

        gray_input = np.expand_dims(gray, axis=-1)
        gray_input = np.expand_dims(gray_input, axis=0)

        pred = model.predict(gray_input, verbose=0)

        digit = int(np.argmax(pred))
        confidence = float(np.max(pred))

        with cols[2]:
            result_placeholder.markdown(
                f"""
                <div class="oneui-result">
                    <div class="oneui-digit-badge">{digit}</div>
                    <div class="oneui-result-text">
                        <div class="label">Predicted digit</div>
                        <div class="conf">{confidence*100:.1f}% confident</div>
                    </div>
                </div>
                <div class="oneui-progress-track">
                    <div class="oneui-progress-fill" style="width:{confidence*100:.1f}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with img_placeholder.container():
                c1, c2, c3 = st.columns([1, 1, 1])
                with c2:
                    st.image(display_img, width=140)
                    st.markdown(
                        '<div class="oneui-thumb-caption">Preprocessed 28×28</div>',
                        unsafe_allow_html=True,
                    )

            # One UI-flavoured probability chart
            fig, ax = plt.subplots(figsize=(6, 2.6))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            colors = ["#4C8DFF" if i == digit else "#33415C" for i in range(10)]
            ax.bar(range(10), pred[0], color=colors, width=0.6, zorder=3)

            ax.set_xticks(range(10))
            ax.set_xlabel("Digit", color="#9AA1AC", fontsize=10)
            ax.set_ylabel("Probability", color="#9AA1AC", fontsize=10)
            ax.tick_params(colors="#9AA1AC", labelsize=9)

            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#2C2D30")
            ax.grid(axis="y", color="#232326", zorder=0)

            plot_placeholder.pyplot(fig, transparent=True)
    else:
        with cols[2]:
            result_placeholder.markdown(
                """
                <div class="oneui-empty">
                    <div class="glyph">⚠️</div>
                    Draw something first, then tap Predict.
                </div>
                """,
                unsafe_allow_html=True,
            )