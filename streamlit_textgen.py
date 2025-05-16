
import streamlit as st
from transformers import pipeline

st.title("Text Generation App")
prompt = st.text_input("Enter your prompt:")
model_name = st.selectbox("Choose a model", ["gpt2", "distilgpt2"])

if st.button("Generate"):
    generator = pipeline("text-generation", model=model_name)
    output = generator(prompt, max_length=50)[0]['generated_text']
    st.write("Generated Text:")
    st.success(output)
