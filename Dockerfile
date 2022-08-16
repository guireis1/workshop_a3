FROM python:3.8-slim-buster AS build

COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN export PYTHONPATH="${PYTHONPATH}:/./app"

COPY ./app /app

#Download BERT 
RUN apt-get update
RUN apt-get install -y curl
RUN curl -L -# "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/distiluse-base-multilingual-cased-v2.zip" -o model.zip
RUN apt install unzip
RUN unzip model.zip -d models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]