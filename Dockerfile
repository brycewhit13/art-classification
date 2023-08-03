# Use Python 3.9 image
FROM python:3.9

# Create working directory
WORKDIR /app

# Copy requirements.txt file
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose port
EXPOSE 8080

# Copy all files for app
COPY . /app

#Run app
CMD ["flask", "run"]