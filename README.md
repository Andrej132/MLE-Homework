# MLE-Homework

To create a docker image:

docker build -t training .

docker build -t inference .

After that, we can run the train.py and main.py files in separate containers with the following commands: 

docker run -v "%cd% \..\model:/app/model:rw" -v "%cd% \..\data:/app/data:rw" training

docker run -v "%cd% \..\model:/app/model:rw" -v "%cd% \..\data:/app/data:rw" inference

