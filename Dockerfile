FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /code

RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cpu
    
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
RUN python -c "import shutil; from app.model import weights_path, WEIGHTS; \
WEIGHTS.parent.mkdir(exist_ok=True); shutil.copy(weights_path(), WEIGHTS)" \
    && rm -rf /root/.cache/huggingface

RUN chown -R user /code
USER user

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]