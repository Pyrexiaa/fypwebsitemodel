FROM python:3.12.8

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN pip install flask_cors

EXPOSE 80

CMD ["python", "app.py"]