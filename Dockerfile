FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Expose port (Railway / Render override via $PORT)
EXPOSE 8000

# Start — use $PORT if set, else 8000
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
