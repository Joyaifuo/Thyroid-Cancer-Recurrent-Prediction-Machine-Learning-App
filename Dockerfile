# Use an official lightweight Python image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt --disable-pip-version-check --progress-bar off 


# Copy the application code
COPY . .

# Expose the Streamlit default port
EXPOSE 8080

# Set environment variables to disable Streamlit's telemetry
ENV STREAMLIT_SERVER_HEADLESS=true

# Run the Streamlit application
CMD ["streamlit", "run", "thyroid_prediction.py", "--server.port=8080", "--server.address=0.0.0.0"]
