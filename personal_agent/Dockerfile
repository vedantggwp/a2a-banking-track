FROM python:3.12-slim

WORKDIR /app
COPY personal_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY personal_agent/ .

EXPOSE 9001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9001"]
