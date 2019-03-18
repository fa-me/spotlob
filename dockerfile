FROM continuumio/miniconda3:latest

LABEL maintainer="fabian.meyer@ise.fraunhofer.de"

# Add the user that will run the app (no need to run as root)
#RUN groupadd -r myuser && useradd -r -g myuser myuser

WORKDIR /app

# missing library
# https://github.com/ContinuumIO/docker-images/issues/49
RUN apt-get update
RUN apt-get install libgl1-mesa-swx11

# Install myapp requirements
COPY environment.yml /app/environment.yml
RUN conda config --add channels conda-forge \
    && conda env create -f environment.yml \
    && rm -rf /opt/conda/pkgs/*

# Install myapp
#COPY . /app/
#RUN chown -R myuser:myuser /app/*

# activate the myapp environment
#ENV PATH /opt/conda/envs/spotlob-env/bin:$PATH