#!/usr/bin/env python
# coding: utf-8
import subprocess
subprocess.run(['pip', 'install', '-Uqq', 'fastai'])
subprocess.run(['pip', 'install', '-Uqq', 'timm'])
from fastai.vision.all import load_learner
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import fonts
import os
from huggingface_hub import InferenceClient

class CustomGradientTheme(Base):
    def __init__(self):
        super().__init__()
        super().set(
            body_background_fill="linear-gradient(to bottom, #E6EAE9, #F3E2D4)",
            body_background_fill_dark="linear-gradient(to bottom, #E6EAE9, #F3E2D4)",
            button_primary_background_fill="#8DBED4",
            button_primary_background_fill_dark="#8DBED4",
            button_secondary_background_fill="#D7D6E4",
            button_secondary_background_fill_dark="#D7D6E4"
        )

custom_theme = CustomGradientTheme()

learn = load_learner('model.pkl')
categories = (
    'Degenerative Infectious Disease',
    'Mediastinal Anomalies',
    'No Finding',
    'Obstructive Pulmonary Disease',
    'Pneumonia'
)

example_images = [
    ["example1.jpg"],
    ["example2.jpeg"],
    ["example3.jpeg"]
]

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

def reset_all():
    return None, None

def explain_diagnosis(prediction):
    if not prediction:
        return "Please analyze an image first to get a diagnosis."
    
    predicted_class = max(prediction, key=prediction.get)
    
    prompt = (
        f"You are a board-certified radiologist providing guidance to another radiologist on how to validate a preliminary diagnosis of {predicted_class} based on a chest x-ray. Provide a step-by-step approach for confirming or ruling out this diagnosis, referencing key radiological signs, differential diagnoses, and any necessary follow-up imaging or tests. Use precise medical terminology and do not include unnecessary explanations of basic concepts a radiologist would already know."
    )
    
    client = InferenceClient(
        provider="together",
        api_key=os.getenv("HF_API_KEY")
    )
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=messages,
        max_tokens=500
    )
    msg = completion.choices[0].message
    answer = msg.content if hasattr(msg, "content") else str(msg)
    answer = answer.replace("<think>", "").replace("</think>", "")

    if not answer.endswith((".", "!", "?")):
        answer += "..."
        
    return answer.strip()

with gr.Blocks(theme=custom_theme) as demo:
    gr.HTML("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    body, .gradio-container, .gradio-container * {
        font-family: 'Inter', sans-serif !important;
    }
    .block, .gr-box, .gr-panel {
        border: none !important;
        border-radius: 8px !important;
    }
    .wrap.svelte-1p9xokt, .wrap.svelte-1p9xokt > div {
        border: none !important;
        border-radius: 8px !important;
    }
    .output-class, .gr-input-label {
        border: none !important;
        border-radius: 8px !important;
    }
    </style>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="xray")
            with gr.Row():
                analyze_button = gr.Button("Analyze", variant="primary")
                reset_button = gr.Button("Reset", variant="secondary")
            gr.Examples(
                examples=example_images,
                inputs=image_input,
                label="examples"
            )
        
        with gr.Column(scale=1):
            label_output = gr.Label(label="results")
            chatbot_output = gr.Textbox(label="explanation", interactive=False)
            explain_button = gr.Button("Explain Results", variant="primary")
    
    analyze_button.click(fn=classify_image, inputs=image_input, outputs=label_output)
    reset_button.click(fn=reset_all, inputs=[], outputs=[image_input, label_output])
    explain_button.click(fn=explain_diagnosis, inputs=label_output, outputs=chatbot_output)

demo.launch(inline=False)