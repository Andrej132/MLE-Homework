FROM python:3.9

WORKDIR /app

COPY . /app/inference
COPY requirements.txt ./

RUN pip install -r requirements.txt

CMD ["python3","inference/main.py"]
