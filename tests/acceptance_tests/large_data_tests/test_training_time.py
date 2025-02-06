import os
import subprocess
import time


def test_run_rasa_train_and_capture_time():
    start_time = time.time()
    result = subprocess.run(
        [
            "rasa",
            "train",
            "--config",
            "tests_deployment/large-data-assistant/config.yml",
            "--endpoints",
            "tests_deployment/large-data-assistant/endpoints.yml",
            "--data",
            "tests_deployment/large-data-assistant/data",
            "--domain",
            "tests_deployment/large-data-assistant/domain.yml",
        ],
        capture_output=True,
        text=True,
    )
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")

    with open("training_time.txt", "w") as f:
        f.write(str(training_time))

    # Set the training time as an environment variable
    os.environ["TRAINING_TIME"] = str(training_time)

    # Optionally, you can check the result of the command
    if result.returncode == 0:
        print("Training completed successfully")
        assert "Your Rasa model is trained and saved at" in result.stderr
    else:
        print("Training failed")
        print(result.stderr)
