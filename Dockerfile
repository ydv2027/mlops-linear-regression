# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Update package lists and upgrade packages to patch vulnerabilities
# The '-y' flag auto-confirms prompts, and 'rm -rf ...' cleans up afterward
# to keep the image size down.
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and artifacts into the container
# The CI job will provide these files.
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/

# Command to run when the container starts
# This will execute the prediction script for verification.
CMD ["python", "src/predict.py"]