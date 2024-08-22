FROM public.ecr.aws/lambda/python:3.9
WORKDIR ${LAMBDA_TASK_ROOT}
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY data/raw/data.csv ./data/raw/
COPY models/similarity.pkl ./models/
COPY templates/ ./templates/
CMD ["app.handler"]