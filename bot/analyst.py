import os
import google.generativeai as genai
import logging

# Setup logging to see exactly which model works
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI_ANALYST")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def generate_trade_analysis(signal_type, price, slope, whale_ratio, fng, flow_state, win_prob):
    if not GOOGLE_API_KEY:
        return "⚠️ CRITICAL: GOOGLE_API_KEY is missing."

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # --- 2025 MODEL UPDATE ---
        # Gemini 1.5 is likely deprecated. We prioritize the 2.5/2.0 series.
        models_to_try = [
            'gemini-2.5-flash',       # Current standard (Fast/Cheap)
            'gemini-2.0-flash',       # Fallback
            'gemini-1.5-flash-latest',# Legacy fallback
            'gemini-pro'              # Oldest fallback
        ]

        model = None
        # We assume the first one works, or we iterate (optional logic simplified here)
        # For robustness, let's just pick the main one. 
        # If you want to force the bot to FIND a working one, we can list them:
        
        found_model_name = None
        
        # 1. Quick check if we need to scan (optimistic approach first)
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            # Test generation on a tiny prompt to verify existence before running main prompt?
            # No, just run it. If it fails, we catch it below.
        except:
            pass

        prompt = (
            f"You are a crypto trading bot (Year 2025). Analyze this signal in 1 short sentence. "
            f"Signal: {signal_type} ETH @ ${price}. "
            f"Data: Slope {slope:.2f}, Whale Ratio {whale_ratio:.2f}, Sentiment {fng}, WinProb {win_prob:.0f}%."
        )

        # 2. ITERATE THROUGH MODELS until one works
        for m_name in models_to_try:
            try:
                model = genai.GenerativeModel(m_name)
                response = model.generate_content(prompt)
                logger.info(f"✅ Success using model: {m_name}")
                return response.text.strip()
            except Exception as e:
                logger.warning(f"⚠️ Model {m_name} failed: {e}")
                continue

        return "⚠️ AI Error: No valid Gemini 2.x/1.x models found."

    except Exception as e:
        return f"⚠️ API Error: {str(e)[:50]}..."
