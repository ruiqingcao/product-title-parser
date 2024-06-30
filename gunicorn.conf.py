# gunicorn.conf.py
timeout = 300
workers = 1
threads = 2
worker_class = 'gthread'
