FROM python:3.11-slim

# Install pip
RUN pip install -U pip
# RUN pipenv
# Assuming you have a requirements.txt file with Flask listed as a dependency
# Make sure to uncomment the next line to include requirements.txt
COPY ["requirements.txt", "./"]

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install Waitress
RUN pip install waitress

# Set working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies
# Uncomment the following line if you prefer to use pipenv
# RUN pipenv install --system --deploy

ENV FLASK_APP=my_flask.py
# Copy application files
COPY ["predict.py", "models/lasso.bin", "./"]

# Expose port 8080
EXPOSE 8080

# Update ENTRYPOINT to use Waitress
ENTRYPOINT ["waitress-serve", "--listen=*:8080", "predict:app"]

# Define environment variable
