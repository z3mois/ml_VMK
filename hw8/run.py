#!/usr/bin/env python3

from sys import argv
import numpy as np
from glob import glob
import shutil
import json
import os
from sklearn.metrics import mean_absolute_error
import traceback


def check_test(data_dir):
    gt_dir = os.path.join(data_dir, 'gt')
    output_dir = os.path.join(data_dir, 'output')

    predictions = np.array(json.load(open(os.path.join(output_dir, 'predictions.json'))))
    target = np.array(json.load(open(os.path.join(gt_dir, 'target.json'))))

    if len(target) != len(predictions):
        res = json.dumps({"status": "FAILED, inconsistent number of predictions", "mae": 20000.})
        if os.environ.get('CHECKER'):
            print(res)
        return res

    mae = mean_absolute_error(target, predictions)

    res = json.dumps({"status": "OK", "mae": mae})
    if os.environ.get('CHECKER'):
        print(res)

    return res


def grade(data_path):
    gradation = {
        5: [0, 2100],
        3: [2100, 2200],
        1: [2200, 2300],
        0: [2300, 1e10]
    }
    student_mark = 0
    maes = []
    results = json.load(open(os.path.join(data_path, 'results.json')))
    for i, result in enumerate(results):
        status = json.loads(result['status'])["status"]
        mae = json.loads(result['status'])["mae"]
        maes.append(mae)

        if status != 'OK':
            mark_info = {'description': 'An error occurred during predictions evaluation', 'mark': 0}
            if os.environ.get('CHECKER'):
                print(json.dumps(mark_info))
            return mark_info
        else:
            current_mark = 0
            for mark, interval in gradation.items():
                if interval[0] < mae <= interval[1]:
                    current_mark = max(mark, current_mark)

            student_mark += current_mark * (1 * i + 1)  # i == 0 for public, 1 for private
    mark_info = {'description': f'OK, mae = {maes}', 'mark': student_mark}
    if os.environ.get('CHECKER'):
        print(json.dumps(mark_info))

    return mark_info


def run_single_test(data_dir, output_dir):
    from awards_prediction import train_model_and_predict

    train_file = os.path.join(data_dir, 'train/train.jsonl')
    test_file = os.path.join(data_dir, 'test/test.jsonl')

    predictions = train_model_and_predict(train_file, test_file)
    json.dump(list(predictions), open(os.path.join(output_dir, "predictions.json"), "w"))


if __name__ == '__main__':
    if os.environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print(f'Usage: {argv[0]} mode data_dir output_dir')
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            check_test(data_dir)
        elif mode == 'grade':
            grade(data_dir)
    else:
        # Script is running locally, run on dir with tests

        tests_dir = argv[1] if len(argv) == 2 else './public_tests/'

        data_dir = glob(os.path.join(tests_dir, '[0-9][0-9]_*_input'))[0]
        gt_dir = glob(os.path.join(tests_dir, '[0-9][0-9]_*_gt'))[0]

        for directory in ["gt", "output"]:
            try:
                os.mkdir(os.path.join(gt_dir, directory))
            except:
                pass

        os.path.join(gt_dir, "gt")
        output_dir = os.path.join(gt_dir, "output")

        shutil.copy(os.path.join(gt_dir, "target.json"), os.path.join(gt_dir, "gt", "target.json"))

        results = []

        try:
            run_single_test(data_dir, output_dir)
        except Exception as ex:
            status = 'Runtime error'
            print(ex)
        else:
            try:
                status = check_test(gt_dir)
            except Exception as ex:
                status = 'Checker error'
                print(ex)
        finally:
            results.append({'status': status})
        if status == 'Runtime error' or status == 'Checker error':
            print("An error occurred")
        else:
            json.dump(results, open(os.path.join(gt_dir, 'results.json'), 'w'))
            res = grade(gt_dir)
            print('Mark:', res['mark'], res['description'])