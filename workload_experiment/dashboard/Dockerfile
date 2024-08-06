# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /usr/src/app

COPY . .

RUN pip3 install dash dash-uploader pandas matplotlib

RUN pip3 install torch scipy

RUN find . -type f -name "*.py" -exec chmod +x {} +

# Make port 8050 available to the world outside this container
EXPOSE 8050

# Run dashboard.py when the container launches
CMD ["python3", "dashboard.py"]
