import argparse
import time
import math
import os
from os import path
import json

from azureml.core.run import Run

CHECKPOINT_FILE_NAME = "model_checkpoint.txt"


def get_logger():
    try:
        return Run.get_submitted_run()
    except Exception:
        return LocalLogger()


class LocalLogger:
    def log(self, key, value):
        print("AML-Log:", key, value)


def evaluate(num_epochs, delay_seconds, model, checkpoint_location):
    run_logger = get_logger()
    start_epoch = 0
    if checkpoint_location:
        try:
            checkpoint_file_path = os.path.join(checkpoint_location, CHECKPOINT_FILE_NAME)
            with open(checkpoint_file_path) as f:
                start_epoch = int(f.readline())
            print("Found a checkpoint. Starting training from epoch: {}".format(start_epoch))
        except Exception as ex:
            print("Exception occurred while trying to read checkpoint file. "
                  "Starting training from epoch 0: {}".format(ex))
    
    model = json.loads(model)
    if model["model_name"] == "model_x":
        x1 = model["x0"]
        x2 = model["x1"]
    elif model["model_name"] == "model_y":
        x1 = model["y0"]
        x2 = model["y1"]
    else:
        print("Invalid HyperParmaters {}".format(model))
        
    for ep in range(start_epoch, num_epochs):
        # x1 = 0, x2 = 0 maximize the result.
        result = math.sqrt(ep * 1000) / (1 + math.pow(x1, 2) + math.pow(x2, 2))
        result /= math.sqrt(num_epochs * 1000)  # Results between 0 and 1
        run_logger.log("result", float(result))
        time.sleep(delay_seconds)

        # Write checkpoints
        output_checkpoint_dir = "./outputs/"
        if not os.path.exists(output_checkpoint_dir):
            os.makedirs(output_checkpoint_dir)
        output_checkpoint_file_path = os.path.join(output_checkpoint_dir, CHECKPOINT_FILE_NAME)
        with open(output_checkpoint_file_path, 'w') as f:
            f.write(str(ep) + "\n")


def main():
    # Get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--delay_seconds', type=float, default=20)
    parser.add_argument('--model', type=str, default='{"model_name": "model_x", "x0": 1, "x1": -0.04776077313177862}')
    parser.add_argument('--resume_from', type=str, default=None)

    args, unknown_args = parser.parse_known_args()

    # log parameters
    run_logger = get_logger()
    run_logger.log("args", args)

    previous_checkpoint_location = args.resume_from

    evaluate(args.num_epochs, args.delay_seconds, args.model, previous_checkpoint_location)


if __name__ == "__main__":
    main()
