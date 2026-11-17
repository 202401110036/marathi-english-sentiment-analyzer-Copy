
import os
import re
import html
import unicodedata
import joblib
import gradio as gr
from scipy.sparse import hstack

# Load trained model and vectorizers
model = joblib.load("sentiment_model.pkl")
word_vectorizer = joblib.load("word_vectorizer.pkl")
char_vectorizer = joblib.load("char_vectorizer.pkl")


# Same preprocessing used during training
def preprocess_text(text):
    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Sentiment prediction
def analyze_sentiment(text):

    if not text or not text.strip():
        return None, "Please enter some text."

    cleaned_text = preprocess_text(text)

    if not cleaned_text:
        return None, "Please enter meaningful text."

    word_features = word_vectorizer.transform([cleaned_text])
    char_features = char_vectorizer.transform([cleaned_text])

    combined_features = hstack([
        word_features,
        char_features
    ]).tocsr()

    probabilities = model.predict_proba(combined_features)[0]

    labels = {
        -1.0: "Negative",
         0.0: "Neutral",
         1.0: "Positive"
    }

    results = {
        labels[float(label)]: float(probability)
        for label, probability in zip(
            model.classes_, probabilities
        )
    }

    prediction = max(results, key=results.get)

    return results, f"Predicted sentiment: {prediction}"


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(),
               title="Marathi-English Sentiment Analyzer") as demo:

    gr.Markdown("""
    # Marathi-English Sentiment Analyzer

    Analyze sentiment in Marathi-English code-mixed text.

    **Model:** Logistic Regression + Combined TF-IDF
    """)

    with gr.Row():

        with gr.Column():
            text_input = gr.Textbox(
                lines=5,
                label="Enter your Marathi-English sentence",
                placeholder="Example: Aaj movie khup chan hota!"
            )

            with gr.Row():
                analyze_button = gr.Button(
                    "Analyze Sentiment",
                    variant="primary"
                )
                clear_button = gr.Button("Clear")

            gr.Examples(
                examples=[
                    ["Aaj movie khup chan hota!"],
                    ["Mala ha movie avadla nahi"],
                    ["Mi ghari jat aahe"],
                    ["Khup chan vatla"],
                    ["Ha movie kharab hota"]
                ],
                inputs=text_input
            )

        with gr.Column():
            sentiment_output = gr.Label(
                label="Sentiment Prediction",
                num_top_classes=3
            )

            message_output = gr.Textbox(
                label="Result",
                interactive=False
            )

    gr.Markdown("""
    ---
    ### Project Information

    - Dataset: L3Cube-MeSent
    - Feature Extraction: Word and Character TF-IDF
    - Algorithm: Logistic Regression
    - Test Accuracy: 62.45%

    *Prediction probabilities are estimates, not guarantees.*
    """)

    analyze_button.click(
        analyze_sentiment,
        inputs=text_input,
        outputs=[sentiment_output, message_output]
    )

    text_input.submit(
        analyze_sentiment,
        inputs=text_input,
        outputs=[sentiment_output, message_output]
    )

    clear_button.click(
        lambda: ("", None, ""),
        outputs=[text_input, sentiment_output, message_output]
    )


# Render server configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
