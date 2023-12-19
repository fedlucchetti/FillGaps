FROM tensorflow/tensorflow:latest-gpu


RUN python3 -m pip install --upgrade pip

# Create directories
RUN mkdir -p /home/FillGaps/data \
    && mkdir -p /home/FillGaps/neuralnet \
    && mkdir -p /home/FillGaps/proc \
    && mkdir -p /home/FillGaps/tools \
    && mkdir -p /home/FillGaps/results

# Copy files into the container
WORKDIR /home/FillGaps
COPY *.py ./
COPY requirements.txt .
COPY setup.py .
# Install requirements
RUN pip3 install -r requirements.txt

# Change permissions
RUN chmod -R ugo=rwx .

# Install the package (assuming there's a setup.py in the current directory)
# RUN pip3 install -e .

# Run as non-root user for better security
RUN useradd -m fillgapsuser
USER fillgapsuser
USER root
RUN pip3 install .
USER fillgapsuser