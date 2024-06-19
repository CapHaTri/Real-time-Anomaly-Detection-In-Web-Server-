# Dockerfile for Streamlit
FROM python:3.8
# Set working directory
WORKDIR /app

# Copy the app.py file into the container at /app
COPY /py/app.py /app
COPY requirements.txt /app
# Install Streamlit and other dependencies
RUN pip install --no-cache-dir -r requirements.txt



# Expose the port for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
