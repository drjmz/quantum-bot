import os
import google.generativeai as genai
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DEGEN_ANALYST")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def generate_trade_analysis(signal_type, price, slope, whale_ratio, fng, flow_state, win_prob):
    # 1. Safety Check
    if not GOOGLE_API_KEY:
        return "⚠️ CRITICAL: GOOGLE_API_KEY is missing."

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # --- ROBUST MODEL LIST (Includes 2025 Future-Proofing) ---
        models_to_try = [
            'gemini-2.5-flash',       # High likelihood in 2025
            'gemini-2.0-flash',       # Fallback
            'gemini-1.5-flash',       # Current Standard
            'gemini-1.5-flash-latest',# Version alias
            'gemini-pro'              # Legacy fallback
        ]

        # 2. Construct the "Degen" Persona Prompt
        prompt = (
            f"You are a reckless crypto degen trader. Summarize this trade in 1 short, hype-filled sentence. "
            f"Signal: {signal_type} ETH @ ${price}. "
            f"Techs: Slope {slope:.2f}, Whales {whale_ratio:.2f}, Sentiment {fng}, WinProb {win_prob:.0f}%."
        )

        # 3. Iterate until one works
        for m_name in models_to_try:
            try:
                model = genai.GenerativeModel(m_name)
                response = model.generate_content(prompt)
                logger.info(f"✅ Degen Analyst used: {m_name}")
                return response.text.strip()
            except Exception as e:
                # Log silently and try next
                continue

        return "⚠️ AI Error: No valid Gemini models found."

    except Exception as e:
        return f"⚠️ API Error: {str(e)[:50]}..."
