# Irpia project suggestion algorithms

## Setup

Launch outside Docker container:

```bash
CELERY_BROCKER=redis://localhost:6379/0 CELERY_BACKEND=redis://localhost:6379/1 celery -A tasks worker --concurrency=1 --loglevel=INFO
```

Or refer to docker-compose.yml documentation at https://github.com/irplab/irpia/tree/dockerize/back#irpia .

