FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Create logs directory inside app
RUN mkdir -p /app/logs

# Railway sets $PORT dynamically — default 8000 for local/Docker
ENV PORT=8000

EXPOSE 8000

CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT
