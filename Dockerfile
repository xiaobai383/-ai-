FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data/cache data/logs output

EXPOSE 7860

CMD ["python", "app.py"]
