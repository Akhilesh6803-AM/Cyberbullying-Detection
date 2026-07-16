FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Copy requirements file first to leverage Docker build cache
COPY requirements.txt /code/requirements.txt

# Install requirements
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Pre-download NLTK data to prevent runtime download failures or permission errors
RUN python -m nltk.downloader stopwords -d /usr/local/share/nltk_data

# Copy the rest of the application files
COPY . .

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Run the Flask app under Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860"]
