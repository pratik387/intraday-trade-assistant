FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install OCI SDK
RUN pip install --no-cache-dir oci

# Copy entrypoint script
COPY oci/docker/entrypoint.py /app/entrypoint.py
RUN chmod +x /app/entrypoint.py

# Create OCI config directory
RUN mkdir -p /root/.oci

# Note: Code and cache will be downloaded at runtime from OCI Object Storage
# This keeps the image small and allows dynamic config changes

# Entrypoint
ENTRYPOINT ["python", "/app/entrypoint.py"]
