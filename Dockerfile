FROM ultralytics/ultralytics:latest-cpu
RUN useradd -m -u 1000 user
WORKDIR /usr/src/app
CMD ["python", "src/main.py"]