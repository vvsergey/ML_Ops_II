FROM python:3.11
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . /app
CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]