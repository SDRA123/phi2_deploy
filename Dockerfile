FROM python:3.11.1-buster

WORKDIR /

COPY requirements.txt .
RUN pip install -r requirements.txt

ADD handler.py .
CMD [ "python", "-u", "/handler.py" ]
