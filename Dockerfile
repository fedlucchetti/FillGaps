FROM tensorflow/tensorflow:latest-gpu

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create directories
RUN mkdir -p /home/Connectonome/FillGaps \
    && mkdir -p /home/Connectonome/FillGaps/data \
    && mkdir -p /home/Connectonome/FillGaps/fillgaps \
    && mkdir -p /home/Connectonome/FillGaps/fillgaps/neuralnet \
    && mkdir -p /home/Connectonome/FillGaps/fillgaps/proc \
    && mkdir -p /home/Connectonome/FillGaps/fillgaps/tools \
    && mkdir -p /home/Connectonome/FillGaps/results \
    && mkdir -p /home/Connectonome/Data

WORKDIR /home/Connectonome/FillGaps
COPY *.py ./
COPY requirements.txt .
COPY setup.py .

# Install requirements
RUN pip3 install -r requirements.txt

# # Change permissions (consider more restrictive permissions)
# RUN chmod -R ugo=rwx .

# Install the package
RUN pip3 install -e .

# Create and switch to a non-root user for better security
RUN useradd -m fillgapsuser
RUN chown -R fillgapsuser:fillgapsuser /home/Connectonome/FillGaps
USER fillgapsuser
RUN pip3 install -e .


# (Optional) Set the entry point or health check here

