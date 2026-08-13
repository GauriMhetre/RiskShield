# Use a slim python image to keep the container size small.
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt FIRST and install dependencies.
# This order is chosen for layer caching: Docker will cache this layer 
# and reuse it in subsequent builds unless requirements.txt itself changes,
# speeding up builds when only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ backend/

# Copy ml directory
# We need to copy ml/ because backend code imports from ml.features and ml.train.
# This includes the model artifact ml/models/model_v1.pkl which is deliberately 
# copied into the image. Note that while this file is in .gitignore (not in git),
# we do NOT put it in .dockerignore so that it gets copied into the image context.
COPY ml/ ml/

# Expose port 8000
EXPOSE 8000

# Run uvicorn using the exec form to start the app.
# --host 0.0.0.0 is REQUIRED because the container must accept connections
# from the host (and other external sources). Default localhost (127.0.0.1) 
# would only allow connections from inside the container itself.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
