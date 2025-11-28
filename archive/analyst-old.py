import google.generativeai as genai
import os

# Configure
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

def generate_trade_analysis(signal_type, price, slope, whale_ratio, sentiment, flow_state, win_prob):
    if not API_KEY:
        return "⚠️ AI Analysis Unavailable (No API Key)"

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Act as a Senior Crypto Quant Analyst. 
        Review this trade setup for ETH/USDT:
        
        - Signal: {signal_type}
        - Price: ${price}
        - Quantum Slope: {slope:.4f} (Trend Strength)
        - Whale Ratio: {whale_ratio:.2f} (Smart Money Positioning)
        - Retail Sentiment: {sentiment}/100 (Fear/Greed)
        - Market Flow: {flow_state}
        - Historical Win Probability: {win_prob:.1f}%
        
        In 2 short sentences, analyze the confluence of these factors. 
        Is this a high-conviction setup? What is the primary risk?
        Do not use financial advice disclaimers. Be direct.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
   
    except Exception as e:
        return f"⚠️ AI Analysis Failed: {str(e)[:50]}"
