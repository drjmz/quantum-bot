FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc

# Install dependencies (Ensure python-telegram-bot is here)
RUN pip install avantis-trader-sdk cdp-sdk pandas numpy scipy requests scikit-learn python-telegram-bot google-generativeai

COPY . .

CMD ["python", "main.py"]
