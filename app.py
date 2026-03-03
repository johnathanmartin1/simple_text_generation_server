# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse

from fastapi.templating import Jinja2Templates

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


# Initialize FastAPI app
app = FastAPI(title="LLM FastAPI Service", version="1.0")

# Model configuration
MODEL_NAME = "gpt2-medium"  # Replace with a larger model if needed
#MODEL_NAME = GPT2LMHeadModel.from_pretrained('gpt2')
device = 0 if torch.cuda.is_available() else -1

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)




templates = Jinja2Templates(directory="static")


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    """Serve the HTML form."""
    return templates.TemplateResponse("index.html", {"request": request})





@app.post("/generate", response_class=HTMLResponse)
async def generate_text(request: Request, user_input: str = Form(...)):
    """Receive user input and generate text."""
    try:
        # Generate text using the LLM
        output = generator(user_input, max_length=100, num_return_sequences=2)
        generated_text = output[0]["generated_text"]
    except Exception as e:
        generated_text = f"Error: {str(e)}"

    # Return the result in the same HTML page
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "user_input": user_input, "generated_text": generated_text}
    )



# Run with: python3 -m uvicorn app:app --reload


