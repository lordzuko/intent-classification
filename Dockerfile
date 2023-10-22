# Stage 1: Build stage
FROM python:3.10.11 as builder

WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools wheel

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application source code
COPY . .
RUN rm requirements.txt

# Stage 2: Production stage
FROM builder

WORKDIR /app

# Copy the installed dependencies from the previous stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the application source code from the previous stage
COPY --from=builder /app/* .

# Expose port 5000
EXPOSE 8080

CMD ["python","server.py"]
