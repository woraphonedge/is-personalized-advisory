# Use a distro-specific tag for stability and security updates
FROM python:3.12-slim

# Install uv globally
RUN pip install uv

# Set the working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV

# Add the virtual environment to the PATH
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy only dependency definition first for better caching
COPY pyproject.toml ./

# Copy the application code
COPY . .

# Install the project and its dependencies
RUN uv pip install .

# Verify installed packages
RUN uv pip list

# Create a non-root user and group
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --no-create-home appuser

# Create cache directory for uv and set permissions
RUN mkdir -p /tmp/uv-cache && \
    chown -R appuser:appgroup /tmp/uv-cache

# Set UV cache directory to a location the non-root user can access
ENV UV_CACHE_DIR=/tmp/uv-cache

# Set ownership to non-root user
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Expose the port the app runs on (matches default PORT=8100)
EXPOSE 8100

# Command to run the application with gunicorn + 4 workers for CPU-bound rebalance
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8100", "--timeout", "60"]
