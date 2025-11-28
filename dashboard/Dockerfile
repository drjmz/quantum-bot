FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for scipy
RUN apt-get update && apt-get install -y gcc

# Install Python libs
RUN pip install streamlit pandas numpy scipy plotly requests

# --- NEW LINE HERE ---
COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
