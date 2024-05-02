FROM python:3.11-slim-buster

# Install pip and pipenv
RUN pip install -U pip
RUN pip install pipenv 

# Set working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies
RUN pipenv install --system --deploy

# Copy application files
COPY ["predict.py", "lasso.bin", "./"]

# Expose port 8080
EXPOSE 8080

# Set entrypoint to use Waitress
CMD ["pipenv", "run", "waitress-serve", "--port=8080", "predict:app"]
