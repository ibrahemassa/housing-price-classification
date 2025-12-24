from prefect import flow, task
import subprocess


@task(retries=2, retry_delay_seconds=5)
def preprocess():
    subprocess.run(
        ["python", "src/preprocess.py"],
        check=True
    )


@task(retries=2, retry_delay_seconds=5)
def train():
    subprocess.run(
        ["python", "src/train.py"],
        check=True
    )


@task(retries=2, retry_delay_seconds=5)
def promote():
    subprocess.run(
        ["python", "src/register_model.py"],
        check=True
    )


@flow(name="housing-price-training-pipeline")
def training_pipeline():
    preprocess()
    train()
    promote()


if __name__ == "__main__":
    training_pipeline()

