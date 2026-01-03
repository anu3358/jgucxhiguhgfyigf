# check_models.py
import google.generativeai as genai
import os

# 1. Put your API Key here directly for this test
api_key = "AIzaSyA3oubW48s2K8ok0TyVSgi6emQVvQwWy7I" 

genai.configure(api_key=api_key)

print("🔎 Checking available models for your API key...\n")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ AVAILABLE: {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")
