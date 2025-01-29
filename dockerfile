# Use Python runtime
FROM python:3.12

# Set the working directory 
WORKDIR /usr/src/app

# Copy the current directory contents into the container 
COPY . .

# Install any needed packages specified in requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080
EXPOSE 8080

# Run the app

CMD ["streamlit", "run", "./GUI.py", "--server.port", "8080"]

