import os
import io
import pickle
import re
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gtts import gTTS


# ================================================================
# CONFIGURATION
# ================================================================

HF_MODEL_ID = "dudekulalokithareshma/english-variety-xlm-r"

APP_DIR = Path(__file__).resolve().parent

CLASSES_FILE = APP_DIR / "models" / "english_variety_classes.pkl"


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="English Variety AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 44px;
        font-weight: 800;
        text-align: center;
        margin-top: 5px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 24px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.35);
        margin-top: 15px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 750;
    }

    .confidence {
        font-size: 20px;
        margin-top: 8px;
    }

    .feature-box {
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-bottom: 10px;
    }

    .model-box {
        padding: 18px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.3);
    }

    .small-note {
        font-size: 14px;
        opacity: 0.8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ================================================================
# HEADER
# ================================================================

st.markdown(
    '<div class="main-title">🌍 English Variety AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'American • British • Indian English Detection using XLM-RoBERTa'
    '</div>',
    unsafe_allow_html=True
)


# ================================================================
# CLASS LABELS
# ================================================================

DEFAULT_CLASSES = [
    "American English",
    "British English",
    "Indian English"
]


@st.cache_resource
def load_classes():

    if CLASSES_FILE.exists():

        try:
            with open(CLASSES_FILE, "rb") as f:
                classes = pickle.load(f)

            if hasattr(classes, "tolist"):
                classes = classes.tolist()

            if isinstance(classes, (list, tuple)) and len(classes) == 3:
                return list(classes)

        except Exception:
            pass

    return DEFAULT_CLASSES


CLASSES = load_classes()


# ================================================================
# LOAD XLM-R FROM HUGGING FACE
# ================================================================

@st.cache_resource(show_spinner=True)
def load_xlmr():

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_ID
    )

    model.eval()

    return tokenizer, model


# ================================================================
# SAFE MODEL LOADING
# ================================================================

try:

    with st.spinner(
        "🤖 Loading English Variety AI XLM-R model..."
    ):
        tokenizer, model = load_xlmr()

    MODEL_READY = True

except Exception as e:

    MODEL_READY = False

    st.error(
        "❌ Could not load the XLM-R model."
    )

    st.code(str(e))

    st.info(
        "Please check that the Hugging Face model repository "
        "is public and available."
    )


# ================================================================
# PREDICTION
# ================================================================

def predict_variety(text):

    if not text or not text.strip():
        raise ValueError("Please enter a sentence.")

    encoded = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    prediction = CLASSES[predicted_index]

    confidence = float(
        probabilities[predicted_index].item()
    ) * 100

    probability_dict = {}

    for i, label in enumerate(CLASSES):

        probability_dict[label] = (
            float(probabilities[i].item()) * 100
        )

    return (
        prediction,
        confidence,
        probability_dict
    )


# ================================================================
# LINGUISTIC KNOWLEDGE
# ================================================================

BRITISH_SPELLINGS = {
    "colour": "color",
    "colours": "colors",
    "favourite": "favorite",
    "favourites": "favorites",
    "centre": "center",
    "centres": "centers",
    "theatre": "theater",
    "theatres": "theaters",
    "organise": "organize",
    "organised": "organized",
    "organising": "organizing",
    "realise": "realize",
    "realised": "realized",
    "travelling": "traveling",
    "cancelled": "canceled",
    "catalogue": "catalog",
    "programme": "program",
    "grey": "gray",
    "neighbour": "neighbor",
    "neighbours": "neighbors",
    "labour": "labor",
    "defence": "defense",
    "licence": "license"
}

AMERICAN_SPELLINGS = {
    value: key
    for key, value in BRITISH_SPELLINGS.items()
}

VOCABULARY_MAP = {

    "American English": {
        "apartment": "flat",
        "elevator": "lift",
        "truck": "lorry",
        "vacation": "holiday",
        "sidewalk": "pavement",
        "gas": "petrol",
        "cell phone": "mobile phone",
        "cookie": "biscuit",
        "fall": "autumn",
        "movie": "film"
    },

    "British English": {
        "flat": "apartment",
        "lift": "elevator",
        "lorry": "truck",
        "holiday": "vacation",
        "pavement": "sidewalk",
        "petrol": "gas",
        "mobile phone": "cell phone",
        "biscuit": "cookie",
        "autumn": "fall",
        "film": "movie"
    },

    "Indian English": {
        "do the needful": "take the necessary action",
        "revert back": "reply",
        "prepone": "move to an earlier time",
        "out of station": "away from the city",
        "pass out": "graduate",
        "updation": "update",
        "discuss about": "discuss"
    }
}


INDIAN_EXPRESSIONS = [
    "do the needful",
    "revert back",
    "prepone",
    "out of station",
    "pass out",
    "updation",
    "discuss about"
]


# ================================================================
# LINGUISTIC ANALYSIS
# ================================================================

def linguistic_analysis(text):

    lower = text.lower()

    vocabulary_hits = []
    spelling_hits = []
    expression_hits = []

    # British spelling clues
    for word in BRITISH_SPELLINGS:

        if re.search(
            r"\b" + re.escape(word) + r"\b",
            lower
        ):
            spelling_hits.append(
                f"{word} → {BRITISH_SPELLINGS[word]}"
            )

    # American spelling clues
    for word in AMERICAN_SPELLINGS:

        if re.search(
            r"\b" + re.escape(word) + r"\b",
            lower
        ):

            spelling_hits.append(
                f"{word} → {AMERICAN_SPELLINGS[word]}"
            )

    # Vocabulary clues
    for variety, words in VOCABULARY_MAP.items():

        for word, alternative in words.items():

            if word in lower:

                vocabulary_hits.append(
                    f"{word} → associated with {variety}"
                )

    # Indian expressions
    for expression in INDIAN_EXPRESSIONS:

        if expression in lower:

            expression_hits.append(
                f"{expression} → Indian English expression"
            )

    # Sentence structure
    word_count = len(text.split())

    if word_count <= 5:
        structure = "Short sentence"
    elif word_count <= 15:
        structure = "Simple/medium-length sentence"
    else:
        structure = "Longer sentence"

    # Punctuation
    punctuation = []

    if "?" in text:
        punctuation.append("Question punctuation")

    if "!" in text:
        punctuation.append("Exclamation punctuation")

    if "," in text:
        punctuation.append("Comma usage")

    if "." in text:
        punctuation.append("Full stop")

    if not punctuation:
        punctuation.append("No strong punctuation clue")

    return {
        "vocabulary": vocabulary_hits,
        "spelling": spelling_hits,
        "expressions": expression_hits,
        "structure": structure,
        "punctuation": punctuation
    }


# ================================================================
# DISPLAY LINGUISTIC ANALYSIS
# ================================================================

def display_linguistic_analysis(text):

    analysis = linguistic_analysis(text)

    st.markdown("---")

    st.markdown("## 🔎 Linguistic Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="feature-box">'
            '<b>📚 Vocabulary</b>'
            '</div>',
            unsafe_allow_html=True
        )

        if analysis["vocabulary"]:

            for item in analysis["vocabulary"]:
                st.write("•", item)

        else:
            st.write(
                "No strong vocabulary clue detected."
            )

        st.markdown(
            '<div class="feature-box">'
            '<b>✏️ Spelling</b>'
            '</div>',
            unsafe_allow_html=True
        )

        if analysis["spelling"]:

            for item in analysis["spelling"]:
                st.write("•", item)

        else:
            st.write(
                "No strong spelling clue detected."
            )

    with col2:

        st.markdown(
            '<div class="feature-box">'
            '<b>💬 Common Expressions</b>'
            '</div>',
            unsafe_allow_html=True
        )

        if analysis["expressions"]:

            for item in analysis["expressions"]:
                st.write("•", item)

        else:
            st.write(
                "No specific regional expression detected."
            )

        st.markdown(
            '<div class="feature-box">'
            '<b>📝 Sentence Structure</b>'
            '</div>',
            unsafe_allow_html=True
        )

        st.write("•", analysis["structure"])

        st.markdown(
            '<div class="feature-box">'
            '<b>🔣 Punctuation / Context</b>'
            '</div>',
            unsafe_allow_html=True
        )

        for item in analysis["punctuation"]:
            st.write("•", item)


# ================================================================
# TEXT TO SPEECH
# ================================================================

def generate_audio(text, variety):

    domain_map = {
        "American English": "com",
        "British English": "co.uk",
        "Indian English": "co.in"
    }

    domain = domain_map.get(
        variety,
        "com"
    )

    if not text or not text.strip():

        raise ValueError(
            "Text is empty."
        )

    audio_buffer = io.BytesIO()

    tts = gTTS(
        text=text.strip(),
        lang="en",
        tld=domain,
        slow=False
    )

    tts.write_to_fp(
        audio_buffer
    )

    audio_buffer.seek(0)

    audio_data = audio_buffer.getvalue()

    if not audio_data:

        raise RuntimeError(
            "gTTS returned empty audio data."
        )

    return audio_data


# ================================================================
# FILE TEXT EXTRACTION
# ================================================================

def extract_uploaded_text(uploaded_file):

    filename = uploaded_file.name.lower()

    # TXT
    if filename.endswith(".txt"):

        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="ignore"
        )

    # DOCX
    if filename.endswith(".docx"):

        try:

            from docx import Document

            document = Document(
                io.BytesIO(
                    uploaded_file.getvalue()
                )
            )

            paragraphs = [
                p.text
                for p in document.paragraphs
                if p.text.strip()
            ]

            return "\n".join(paragraphs)

        except Exception as e:

            raise RuntimeError(
                f"DOCX extraction failed: {e}"
            )

    # PDF
    if filename.endswith(".pdf"):

        try:

            from pypdf import PdfReader

            reader = PdfReader(
                io.BytesIO(
                    uploaded_file.getvalue()
                )
            )

            pages = []

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    pages.append(text)

            return "\n".join(pages)

        except Exception as e:

            raise RuntimeError(
                f"PDF extraction failed: {e}"
            )

    raise ValueError(
        "Unsupported file type. "
        "Please upload TXT, DOCX, or PDF."
    )


# ================================================================
# ENGLISH VARIETY CONVERSION
# ================================================================

def replace_phrases(text, replacements):

    result = text

    # Longer phrases first
    sorted_items = sorted(
        replacements.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for source, target in sorted_items:

        pattern = r"\b" + re.escape(source) + r"\b"

        result = re.sub(
            pattern,
            target,
            result,
            flags=re.IGNORECASE
        )

    return result


def convert_text(text, source_variety, target_variety):

    if source_variety == target_variety:
        return text

    result = text

    # British → American
    if (
        source_variety == "British English"
        and target_variety == "American English"
    ):

        result = replace_phrases(
            result,
            BRITISH_SPELLINGS
        )

        result = replace_phrases(
            result,
            {
                "flat": "apartment",
                "lift": "elevator",
                "lorry": "truck",
                "holiday": "vacation",
                "pavement": "sidewalk",
                "petrol": "gas"
            }
        )

    # American → British
    elif (
        source_variety == "American English"
        and target_variety == "British English"
    ):

        result = replace_phrases(
            result,
            AMERICAN_SPELLINGS
        )

        result = replace_phrases(
            result,
            {
                "apartment": "flat",
                "elevator": "lift",
                "truck": "lorry",
                "vacation": "holiday",
                "sidewalk": "pavement",
                "gas": "petrol"
            }
        )

    # Indian → neutral/international equivalents
    elif source_variety == "Indian English":

        indian_replacements = {
            "do the needful": "take the necessary action",
            "revert back": "reply",
            "prepone": "move to an earlier time",
            "out of station": "away from the city",
            "pass out": "graduate",
            "updation": "update",
            "discuss about": "discuss"
        }

        result = replace_phrases(
            result,
            indian_replacements
        )

    # Convert to Indian style
    elif target_variety == "Indian English":

        result = replace_phrases(
            result,
            {
                "take the necessary action":
                    "do the needful",
                "reply":
                    "revert back",
                "move to an earlier time":
                    "prepone"
            }
        )

    return result


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    st.markdown("## 🌍 English Variety AI")

    st.markdown(
        "### Supported varieties"
    )

    st.write("🇺🇸 American English")
    st.write("🇬🇧 British English")
    st.write("🇮🇳 Indian English")

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.write(
        "XLM-RoBERTa"
    )

    st.caption(
        "Model loaded from Hugging Face"
    )

    st.markdown("---")

    st.markdown("### 📌 Features")

    st.write("✓ Variety detection")
    st.write("✓ Confidence score")
    st.write("✓ Linguistic analysis")
    st.write("✓ File upload")
    st.write("✓ Variety conversion")
    st.write("✓ Pronunciation")


# ================================================================
# MODEL STATUS
# ================================================================

if MODEL_READY:

    st.success(
        "🤖 XLM-R model ready"
    )

else:

    st.warning(
        "⚠️ XLM-R model is not currently available."
    )


# ================================================================
# MAIN INPUT
# ================================================================

st.markdown("## ✍️ Enter Your Sentence")

text_input = st.text_area(
    "Type an English sentence below:",
    height=150,
    placeholder=(
        "Example: "
        "The colour of my favourite car is parked near the lift."
    ),
    key="main_text"
)


# ================================================================
# FILE UPLOAD
# ================================================================

st.markdown("### 📁 Or Upload a File")

uploaded_file = st.file_uploader(
    "Upload TXT, DOCX, or PDF",
    type=["txt", "docx", "pdf"]
)

if uploaded_file is not None:

    try:

        extracted_text = extract_uploaded_text(
            uploaded_file
        )

        if extracted_text.strip():

            st.success(
                f"✅ Text extracted from {uploaded_file.name}"
            )

            st.text_area(
                "Extracted text",
                value=extracted_text,
                height=180,
                key="extracted_preview"
            )

            if not text_input.strip():

                text_input = extracted_text

        else:

            st.warning(
                "The uploaded file did not contain readable text."
            )

    except Exception as e:

        st.error(str(e))


# ================================================================
# ANALYZE BUTTON
# ================================================================

analyze_button = st.button(
    "🔍 Analyze English Variety",
    type="primary",
    use_container_width=True
)


# ================================================================
# ANALYSIS
# ================================================================

if analyze_button:

    if not MODEL_READY:

        st.error(
            "The XLM-R model is not ready."
        )

    elif not text_input.strip():

        st.warning(
            "Please enter a sentence or upload a text file."
        )

    else:

        try:

            with st.spinner(
                "🔎 Analyzing your sentence..."
            ):

                (
                    prediction,
                    confidence,
                    probabilities
                ) = predict_variety(
                    text_input
                )

            # Store results
            st.session_state["prediction"] = prediction
            st.session_state["confidence"] = confidence
            st.session_state["probabilities"] = probabilities
            st.session_state["analyzed_text"] = text_input

            # ====================================================
            # RESULT
            # ====================================================

            st.markdown("---")

            st.markdown("## 🎯 Prediction")

            flag_map = {
                "American English": "🇺🇸",
                "British English": "🇬🇧",
                "Indian English": "🇮🇳"
            }

            flag = flag_map.get(
                prediction,
                "🌍"
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-title">
                        {flag} {prediction}
                    </div>
                    <div class="confidence">
                        Confidence: <b>{confidence:.2f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ====================================================
            # CLASS PROBABILITIES
            # ====================================================

            st.markdown(
                "### 📊 Class Probabilities"
            )

            probability_cols = st.columns(3)

            for column, label in zip(
                probability_cols,
                CLASSES
            ):

                with column:

                    value = probabilities.get(
                        label,
                        0.0
                    )

                    st.metric(
                        label,
                        f"{value:.2f}%"
                    )

                    st.progress(
                        min(
                            max(
                                value / 100,
                                0.0
                            ),
                            1.0
                        )
                    )

            # ====================================================
            # LINGUISTIC ANALYSIS
            # ====================================================

            display_linguistic_analysis(
                text_input
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# ================================================================
# PRONUNCIATION
# ================================================================

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]

    analyzed_text = st.session_state.get(
        "analyzed_text",
        text_input
    )

    st.markdown("---")

    st.markdown("## 🔊 Pronunciation")

    st.write(
        f"Detected variety: **{prediction}**"
    )

    if st.button(
        f"🔊 Generate {prediction} Voice",
        use_container_width=True,
        key="generate_detected_voice"
    ):

        try:

            with st.spinner(
                "Generating pronunciation..."
            ):

                audio_data = generate_audio(
                    analyzed_text,
                    prediction
                )

            if audio_data:

                st.session_state[
                    "voice_audio"
                ] = audio_data

                st.session_state[
                    "voice_variety"
                ] = prediction

                st.success(
                    f"✅ {prediction} pronunciation generated."
                )

        except Exception as e:

            st.error(
                "❌ Could not generate pronunciation."
            )

            st.exception(e)

    if "voice_audio" in st.session_state:

        audio_data = st.session_state[
            "voice_audio"
        ]

        voice_variety = st.session_state.get(
            "voice_variety",
            prediction
        )

        st.markdown(
            f"### 🎧 {voice_variety} Pronunciation"
        )

        st.audio(
            audio_data,
            format="audio/mpeg",
            autoplay=False
        )

        st.download_button(
            label="⬇️ Download Pronunciation MP3",
            data=audio_data,
            file_name=(
                "english_variety_pronunciation.mp3"
            ),
            mime="audio/mpeg",
            use_container_width=True,
            key="download_detected_voice"
        )

        st.caption(
            "▶️ Press the play button above to hear the pronunciation."
        )


# ================================================================
# COMPARE PRONUNCIATIONS
# ================================================================

if "prediction" in st.session_state:

    analyzed_text = st.session_state.get(
        "analyzed_text",
        text_input
    )

    st.markdown("---")

    st.markdown(
        "## 🌎 Compare Pronunciations"
    )

    pronunciation_options = [
        "American English",
        "British English",
        "Indian English"
    ]

    selected_variety = st.selectbox(
        "Choose a variety:",
        pronunciation_options,
        key="comparison_variety"
    )

    if st.button(
        f"🔊 Generate {selected_variety} Audio",
        use_container_width=True,
        key="comparison_audio_button"
    ):

        try:

            with st.spinner(
                f"Generating {selected_variety} audio..."
            ):

                comparison_audio = generate_audio(
                    analyzed_text,
                    selected_variety
                )

            st.audio(
                comparison_audio,
                format="audio/mpeg"
            )

            st.download_button(
                label="⬇️ Download Audio",
                data=comparison_audio,
                file_name=(
                    selected_variety
                    .lower()
                    .replace(" ", "_")
                    + "_pronunciation.mp3"
                ),
                mime="audio/mpeg",
                use_container_width=True,
                key="comparison_download"
            )

            st.success(
                f"✅ {selected_variety} audio ready."
            )

        except Exception as e:

            st.error(
                f"❌ Audio generation failed: {e}"
            )


# ================================================================
# CONVERSION
# ================================================================

if "prediction" in st.session_state:

    analyzed_text = st.session_state.get(
        "analyzed_text",
        text_input
    )

    detected_variety = st.session_state[
        "prediction"
    ]

    st.markdown("---")

    st.markdown(
        "## 🔄 English Variety Conversion"
    )

    conversion_target = st.selectbox(
        "Convert this sentence to:",
        [
            "American English",
            "British English",
            "Indian English"
        ],
        key="conversion_target"
    )

    if st.button(
        "🔄 Convert Sentence",
        use_container_width=True,
        key="convert_button"
    ):

        converted = convert_text(
            analyzed_text,
            detected_variety,
            conversion_target
        )

        st.markdown(
            f"### {conversion_target}"
        )

        st.text_area(
            "Converted text",
            value=converted,
            height=120,
            key="converted_result"
        )

        if converted == analyzed_text:

            st.info(
                "No direct conversion change was detected "
                "for this sentence."
            )


# ================================================================
# MODEL INFORMATION
# ================================================================

st.markdown("---")

st.markdown("## 🤖 Model Information")

st.markdown(
    f"""
    <div class="model-box">

    <b>Model:</b> XLM-RoBERTa<br><br>

    <b>Hugging Face repository:</b>
    {HF_MODEL_ID}<br><br>

    <b>Classes:</b>
    American English • British English • Indian English<br><br>

    <b>Task:</b>
    Three-class English variety classification<br><br>

    <b>Additional analysis:</b>
    Vocabulary, spelling, grammar-related clues,
    expressions, sentence structure, punctuation,
    and contextual information.

    </div>
    """,
    unsafe_allow_html=True
)


# ================================================================
# FOOTER
# ================================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; opacity:0.75;">
    🌍 English Variety AI<br>
    American • British • Indian English
    </div>
    """,
    unsafe_allow_html=True
)



