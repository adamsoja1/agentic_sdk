
# Use official small Python runtime image
FROM python:3.11-slim

WORKDIR /app

# Install build deps and pip packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Copy source
COPY . .
RUN pip install litellm[proxy]
ENV PYTHONUNBUFFERED=1

CMD ["python3","-m", "src.discord.app"]

