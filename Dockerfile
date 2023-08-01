FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirement.txt .

RUN pip install -r requirement.txt

COPY tasks.py level_french_labels.py ./
COPY models ./models

RUN useradd -ms /bin/bash celery

RUN chown -R celery:celery /app /tmp

USER celery

CMD ["python", "-m", "celery", "-A", "tasks", "worker", "--loglevel=info", "--pool=solo"]