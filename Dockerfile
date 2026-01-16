FROM python:3.11-slim

WORKDIR /app

COPY code/streamlit/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/streamlit/ /app

EXPOSE 8501

CMD ["streamlit", "run", "start.py", "--server.port=8501", "--server.address=0.0.0.0"]
