FROM insightface-cuda

COPY . .

CMD ["python", "main.py"]