FROM python:3.12-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Run the script
CMD ["python", "response_a.py"]

