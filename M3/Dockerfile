# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt into the container at /app
COPY requirements.txt requirements.txt

# Install any needed packages
RUN pip install -r requirements.txt

# Copy the model.joblib and app.py into the container
COPY model.joblib model.joblib
COPY app.py app.py

# Expose the port the app runs on
EXPOSE 5000

# Run flask app when the container launches
CMD ["python", "app.py"]
