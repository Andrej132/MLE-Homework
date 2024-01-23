FROM python:3.9

WORKDIR /app

COPY . /app/training
COPY requirements.txt ./

RUN pip install -r requirements.txt

CMD ["python3", "training/train.py"]