FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY ML_Final_Final.csv .
COPY . .
CMD ["python", "RandomForest.py"]
