FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy production requirements and install
COPY --chown=user requirements-prod.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

# Copy application files
COPY --chown=user templates/ ./templates/
COPY --chown=user static/ ./static/
COPY --chown=user app.py .
COPY --chown=user model.h5 .

# Expose port (Hugging Face Spaces expects 7860 by default)
EXPOSE 7860

# Run Flask using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
