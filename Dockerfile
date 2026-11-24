FROM python:3.12-slim

WORKDIR /code

RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
RUN python -c "from app.model import weights_path; weights_path()"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
