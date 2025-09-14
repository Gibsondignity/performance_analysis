# Use Python 3.11 slim image (Debian-based, supports OpenMP)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (needed for sklearn, numpy, pandas, matplotlib)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Expose port
EXPOSE 8000

# Run server (you can override this in docker-compose)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
