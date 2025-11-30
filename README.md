# mlops_hw1_Kublashvili_Tatsiana
Home work 1 for MLOps includes work with next instruments: git + DVC + MLFlow

## 1. Цель проекта

Задание направлено на построение минимального, но полноценного MLOps-ĸонтура, обеспечивающего:
- воспроизводимость эĸспериментов;
- ĸонтроль версий данных и моделей;
- автоматизацию процесса обучения и оценĸи модели;
- доĸументирование процесса через DVC и MLflow.

## 2. Как запустить

Все команды выполняются из корня репозитория (`mlops_hw1_Kublashvili_Tatsiana`):

```bash
# 1. Установить зависимости (при необходимости)
pip install -r [requirements.txt](http://_vscodecontentref_/1)

# 2. Убедиться, что настроен remote для DVC (локальный ./dvc-storage)
dvc remote list

# 3. Воспроизвести весь пайплайн (prepare + train)
dvc repro -v

# 4. Отправить данные и модель в удалённое хранилище DVC
dvc push -v

# 5. (Опционально) восстановить данные/модель на другой машине
dvc pull -v
```

## 3. Краткое описание пайплайна

Пайплайн описан в файле dvc.yaml и состоит из двух стадий:

- prepare

  - cmd: python src/prepare.py
  - входы: src/prepare.py, params.yaml
  - выходы: data/processed/train.csv, data/processed/test.csv
  - назначение: загрузка датасета Iris, разбиение на train/test согласно параметрам в params.yaml.

- train

  - cmd: python src/train.py
  - входы: src/train.py, data/processed/train.csv, params.yaml
  - выходы: модель models/model.pkl, метрики models/metrics.json
  - назначение: обучение модели (Ridge/Logistic Regression) с параметрами из params.yaml,
  - логирование метрик и модели в MLflow и версионирование артефактов через DVC.
  - Связь стадий: train зависит от результатов prepare, поэтому при изменении данных или параметров
  - команда dvc repro автоматически пересобирает нужные шаги.

## 4. Где смотреть UI MLflow

Логи MLflow сохраняются локально в директории mlruns/.

Чтобы запустить UI MLflow:
```bash
mlflow ui --backend-store-uri "file://$(pwd)/mlruns" --host 127.0.0.1 --port 5000
```

После запуска интерфейс будет доступен в браузере по адресу:
http://127.0.0.1:5000

В UI можно просматривать эксперименты, параметры, метрики и загруженные модели. 
