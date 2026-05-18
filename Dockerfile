FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gradio==5.34.0

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
