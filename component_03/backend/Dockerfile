FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY c03_engine ./c03_engine
ENV C03_EXPORT=/export
EXPOSE 8000
CMD ["sh", "-c", "python -m c03_engine --export ${C03_EXPORT} --mode ${C03_MODE:-live} --host 0.0.0.0 --port 8000"]
