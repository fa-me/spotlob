FROM continuumio/miniconda3:latest

LABEL maintainer="fabian.meyer@ise.fraunhofer.de"

WORKDIR /app

COPY environment.yml /app/environment.yml
RUN conda config --add channels conda-forge \
    && conda env create -f environment.yml \
    && rm -rf /opt/conda/pkgs/*
