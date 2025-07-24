# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Allow statements and log messages to be sent straight to the terminal
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Run the web service on container startup.
# Use gunicorn for production deployments.
# The --bind flag is used to listen on all interfaces,
# and the --workers flag is used to specify the number of worker processes.
# The --timeout flag is used to specify the timeout for worker processes.
CMD exec streamlit run app.py --server.port $PORT --server.address 0.0.0.0
