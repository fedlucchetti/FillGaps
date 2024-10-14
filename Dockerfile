FROM tensorflow/tensorflow:latest-gpu

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Define a base directory for the user
ARG BASE_DIR=/home/mrsinetuser

# Create directories using the base directory variable
RUN mkdir -p $BASE_DIR/fillgaps/neuralnet \
             $BASE_DIR/fillgaps/proc \
             $BASE_DIR/tools \
             $BASE_DIR/results \
             $BASE_DIR/experiments \
             $BASE_DIR/data

WORKDIR $BASE_DIR

COPY *.py ./
COPY requirements.txt .
COPY setup.py .

# Install requirements
RUN pip3 install -r requirements.txt

# Create and switch to a non-root user for better security
RUN useradd -m mrsinetuser
RUN chown -R mrsinetuser:mrsinetuser $BASE_DIR

# Set the user for the rest of the Dockerfile
USER mrsinetuser
