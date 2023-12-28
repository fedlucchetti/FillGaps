FROM tensorflow/tensorflow:latest-gpu

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Define a base directory variable
ARG BASE_DIR=/Connectome/Analytics

# Create directories using the base directory variable
RUN mkdir -p $BASE_DIR/fillgaps/neuralnet \
             $BASE_DIR/fillgaps/proc \
             $BASE_DIR/tools \
             $BASE_DIR/results \
             $BASE_DIR/experiments \
             /Connectome/Data

WORKDIR $BASE_DIR

COPY *.py ./
COPY requirements.txt .
COPY setup.py .

# Install requirements
RUN pip3 install -r requirements.txt

# # Change permissions (consider more restrictive permissions)
# RUN chmod -R ugo=rwx .


# Create and switch to a non-root user for better security
RUN useradd -m analyticsuser
RUN chown -R analyticsuser:analyticsuser $BASE_DIR
USER analyticsuser


# (Optional) Set the entry point or health check here

