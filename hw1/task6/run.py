#!/usr/bin/env python3

from json import load, dump, dumps
from glob import glob
from os import environ
from os.path import join
from sys import argv, exit
import os
import re


def run_single_test(data_dir, output_dir):
    from task6 import check
    with open(join(data_dir, 'input.txt')) as f:
        check(f.read().strip(), join(output_dir, 'file.txt'))


def check_test(data_dir):
    output_dir = os.path.join(data_dir, 'output')
    gt_dir = os.path.join(data_dir, 'gt')

    with open(join(output_dir, 'file.txt')) as f:
        output = f.read().strip()

    with open(join(gt_dir, 'file.txt')) as f:
        gt = f.read().strip()

    if output == gt:
        res = f'Ok'
    else:
        res = f'Files are not the same'

    if environ.get('CHECKER'):
        print(res)

    return res


def grade(data_dir):
    results = load(open(join(data_dir, 'results.json')))
    ok_count = 0

    for result in results:
        if result['status'] == 'Ok':
            ok_count += 1

    if ok_count == 3:
        mark = 3
    else:
        mark = 0

    total_count = len(results)
    description = '%02d/%02d' % (ok_count, total_count)

    res = {'description': description, 'mark': mark}

    if environ.get('CHECKER'):
        print(dumps(res))

    return res


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print('Usage: %s mode data_dir output_dir' % argv[0])
            exit(0)

        mode, data_dir, output_dir = argv[1], argv[2], argv[3]

        if mode == 'run_single_test':
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            check_test(data_dir)
        elif mode == 'grade':
            grade(data_dir)
    else:
        # Script is running locally
        if len(argv) != 2:
            print(f'Usage: {argv[0]} tests_dir')
            exit(0)

        from glob import glob
        from json import dump
        from re import sub
        from time import time
        from traceback import format_exc
        from os import makedirs
        from os.path import basename, exists
        from shutil import copytree

        tests_dir = argv[1]

        results = []
        for input_dir in sorted(glob(join(tests_dir, '[0-9][0-9]_*_input'))):
            output_dir = sub('input$', 'check', input_dir)
            run_output_dir = join(output_dir, 'output')
            makedirs(run_output_dir, exist_ok=True)
            gt_src = sub('input$', 'gt', input_dir)
            gt_dst = join(output_dir, 'gt')
            if not exists(gt_dst):
                copytree(gt_src, gt_dst)

            try:
                start = time()
                run_single_test(input_dir, run_output_dir)
                end = time()
                running_time = end - start
            except:
                status = 'Runtime error'
                traceback = format_exc()
            else:
                try:
                    status = check_test(output_dir)
                except:
                    status = 'Checker error'
                    traceback = format_exc()

            test_num = basename(input_dir)[:2]
            if status == 'Runtime error' or status == 'Checker error':
                print(test_num, status, '\n', traceback)
                results.append({'status': status})
            else:
                results.append({'status': status})

        dump(results, open(join(tests_dir, 'results.json'), 'w'))
        res = grade(tests_dir)
        print('Mark:', res['mark'], res['description'])
