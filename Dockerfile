FROM python:3.9-slim

RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip
RUN pip install mlflow
EXPOSE 5000
CMD ["mlflow", "server", "--port", "5000", "--host", "localhost"]