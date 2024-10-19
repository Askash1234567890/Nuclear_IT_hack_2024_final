FROM python:3.8-slim
WORKDIR /app
ADD requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT ["python", "app.py"]
EXPOSE 5000
