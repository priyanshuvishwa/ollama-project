from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "MBZUAI/LaMini-GPT-124M"  # Replace with desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "MBZUAI/LaMini-GPT-124M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

st.title("LaMini-LM Chatbot")

user_input = st.text_input("You:", "")

if user_input:
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=500, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(f"Bot: {response}")
