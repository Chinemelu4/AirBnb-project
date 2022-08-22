FROM python:3.9.7-slim

RUN pip install -U pip
 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

COPY [ "predict.py", "lr.bin", "./" ]

EXPOSE 9696

ENTRYPOINT [ "waitress-serve", "--listen=*:8000 app:app", "predict:app" ]