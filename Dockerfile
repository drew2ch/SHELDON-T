FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
COPY app/ ./app/

ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 8000
CMD ["uvicorn", "app.app:APP", "--host", "0.0.0.0", "--port", "8000"]
