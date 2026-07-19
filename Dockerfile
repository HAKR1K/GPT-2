# Use official lightweight Python image
FROM python:3.10-slim

# Set up environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user named "user" with UID 1000 (recommended for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# Set home directory environment variable
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory inside user home
WORKDIR $HOME/app

# Copy the rest of the application files and ensure they are owned by "user"
COPY --chown=user . $HOME/app

# Expose port (7860 is the default port for Hugging Face Spaces)
EXPOSE 7860

# Run the FastAPI server using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
