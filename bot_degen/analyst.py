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

        # DYNAMIC PROMPT BASED ON STRATEGY
        strategy_context = (
            "You are a contrarian trading algorithm. "
            "Your strategy is to BUY when sentiment is FEAR (reversal) and SELL when sentiment is GREED. "
            "If slope is negative but sentiment is low, view this as a potential 'Long Squeeze' setup. "
            "If slope is positive but sentiment is extreme, view this as a top."
        )

        prompt = (
            f"{strategy_context}\n"
            f"Analyze this {signal_type} signal for ETH in 1 sentence. "
            f"Current Data: Price ${price}, Slope {slope:.2f} (Trend), "
            f"Whale Ratio {whale_ratio:.2f} (>1.2 is bullish accumulation), "
            f"Sentiment {fng} (0-25 is Extreme Fear/Buy Zone). "
            f"Win Probability: {win_prob:.1f}%."
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
