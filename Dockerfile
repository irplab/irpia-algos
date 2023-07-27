FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirement.txt .

RUN pip install -r requirement.txt

COPY tasks.py level_french_labels.py ./
COPY models ./models

CMD ["python", "-m", "celery", "-A", "tasks", "worker", "--loglevel=info", "--pool=solo"]