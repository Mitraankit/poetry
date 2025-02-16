import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model
model_name = "Llama-2-7b-chat-finetune-poetry"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Streamlit UI
st.title("Personalized Poetry Generator")
st.write("Enter details to generate a personalized poem.")

# User inputs
theme = st.text_input("Enter the theme of the poem (e.g., Love, Rain, Hope)")
style = st.text_input("Enter the style of the poem (e.g., Shakespearean, Haiku, Modern)")
keywords = st.text_input("Enter keywords to include in the poem (comma-separated, optional)")

if st.button("Generate Poetry"):
    with st.spinner("Generating poetry..."):
        # Constructing the prompt
        prompt = f"Write a poem about {theme} in a {style} style"
        if keywords:
            prompt += f" using these keywords: {keywords}"
        
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
        result = pipe(f"[INST] {prompt} [/INST]")
        poetry = result[0]['generated_text']
        
        st.subheader("Generated Poetry:")
        st.write(poetry)
