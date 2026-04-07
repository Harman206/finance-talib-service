FROM python:3.11-slim

# Install TA-Lib C library (required before the Python wrapper)
RUN apt-get update && apt-get install -y \
    wget build-essential \
    && wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar -xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4 && ./configure --prefix=/usr && make && make install \
    && cd .. && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
