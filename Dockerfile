FROM python:3.9-slim

# Install necessary system dependencies for pycld3
RUN apt-get update && apt-get install -y \
    libicu-dev \
    libprotobuf-dev \
    protobuf-compiler && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create and switch to a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy only the requirements.txt file to install dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the specific files and folders you need
COPY --chown=user app.py /app/app.py
COPY --chown=user model /app/model
COPY --chown=user templates /app/templates

# Set the Transformers cache directory
ENV TRANSFORMERS_CACHE=/tmp/cache

# Expose the port
EXPOSE 7860

# Start the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]