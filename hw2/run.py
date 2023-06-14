#!/usr/bin/env python3

import os
import numpy as np
import time
import pickle

from json import load, dump, dumps
from glob import glob
from os import environ
from sys import argv, exit
from re import sub
from traceback import format_exc
from shutil import copytree

F1_TESTS_NUM = 7
F2_TESTS_NUM = 5
F3_TESTS_NUM = 7
F4_TESTS_NUM = 4
F5_TESTS_NUM = 6
F6_TESTS_NUM = 3



def run_single_test(data_dir, output_dir):
    import functions_vectorised
    import functions

    if os.path.exists(os.path.join(data_dir, 'task1_vectorised')): 
        for i in range(F1_TESTS_NUM):
            x = np.load(os.path.join(data_dir, f'input_{i}', 'X.npy'))
            tick = time.time()
            res = functions_vectorised.sum_non_neg_diag(x)
            res_time = time.time() - tick
            data = {'res': res, 'time': res_time}
    
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'wb') as f:
                pickle.dump(data, f)
        open(os.path.join(output_dir, 'task1_vectorised'), 'a').close()
    
    elif os.path.exists(os.path.join(data_dir, 'task1')):
        for i in range(F1_TESTS_NUM):
            x = np.load(os.path.join(data_dir, f'input_{i}', 'X.npy'))
            res = functions.sum_non_neg_diag(x.tolist())
            data = {'res': res}
    
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'wb') as f:
                pickle.dump(data, f)
         
        open(os.path.join(output_dir, 'task1'), 'a').close()
        
    
    elif os.path.exists(os.path.join(data_dir, 'task2_vectorised')): 
        for i in range(F2_TESTS_NUM):
            x = np.load(os.path.join(data_dir, f'input_{i}', 'x.npy'))
            y = np.load(os.path.join(data_dir, f'input_{i}', 'y.npy'))
            tick = time.time()
            res = functions_vectorised.are_multisets_equal(x, y)
            res_time = time.time() - tick
            data = {'res': res, 'time': res_time}
    
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'wb') as f:
                pickle.dump(data, f)
        
        open(os.path.join(output_dir, 'task2_vectorised'), 'a').close()

    elif os.path.exists(os.path.join(data_dir, 'task2')): 
        for i in range(F2_TESTS_NUM):
            x = np.load(os.path.join(data_dir, f'input_{i}', 'x.npy'))
            y = np.load(os.path.join(data_dir, f'input_{i}', 'y.npy'))
            res = functions.are_multisets_equal(x.tolist(), y.tolist())
            data = {'res': res}
    
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'wb') as f:
                pickle.dump(data, f)
        
        open(os.path.join(output_dir, 'task2'), 'a').close()
    
    elif os.path.exists(os.path.join(data_dir, 'task3_vectorised')): 
        for i in range(F3_TESTS_NUM):
            x = np.load(os.path.join(data_dir, f'input_{i}', 'x.npy'))
            tick = time.time()
            res = functions_vectorised.max_prod_mod_3(x)
            res_time = time.time() - tick
            data = {'res': res, 'time': res_time}
    
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'wb') as f:
                pickle.dump(data, f)
        
        open(os.path.join(output_dir, 'task3_vectorised'), 'a').close()

    elif os.path.exists(os.path.join(data_dir, 'task3')): 
        for i in range(F3_TESTS_NUM):
            x = np.load(os.path.join(data_dir, f'input_{i}', 'x.npy'))
            res = functions.max_prod_mod_3(x.tolist())
            data = {'res': res}
    
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'wb') as f:
                pickle.dump(data, f)
        
        open(os.path.join(output_dir, 'task3'), 'a').close()
    
    elif os.path.exists(os.path.join(data_dir, 'task4_vectorised')): 
        for i in range(F4_TESTS_NUM):
            image = np.load(os.path.join(data_dir, f'input_{i}', 'image.npy'))
            weights = np.load(os.path.join(data_dir, f'input_{i}', 'weights.npy'))
            tick = time.time()
            res = functions_vectorised.convert_image(image, weights)
            res_time = time.time() - tick
            data = {'res': res, 'time': res_time}
    
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'wb') as f:
                pickle.dump(data, f)
        
        open(os.path.join(output_dir, 'task4_vectorised'), 'a').close()

    elif os.path.exists(os.path.join(data_dir, 'task4')): 
        for i in range(F4_TESTS_NUM):
            image = np.load(os.path.join(data_dir, f'input_{i}', 'image.npy'))
            weights = np.load(os.path.join(data_dir, f'input_{i}', 'weights.npy'))
            res = functions.convert_image(image.tolist(), weights.tolist())
            data = {'res': res}
    
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'wb') as f:
                pickle.dump(data, f)
        
        open(os.path.join(output_dir, 'task4'), 'a').close()
    
    if os.path.exists(os.path.join(data_dir, 'task5_vectorised')): 
        for i in range(F5_TESTS_NUM):
            x = np.load(os.path.join(data_dir, f'input_{i}', 'x.npy'))
            y = np.load(os.path.join(data_dir, f'input_{i}', 'y.npy'))
            tick = time.time()
            res = functions_vectorised.rle_scalar(x, y)
            res_time = time.time() - tick
            data = {'res': res, 'time': res_time}
    
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'wb') as f:
                pickle.dump(data, f)
        
        open(os.path.join(output_dir, 'task5_vectorised'), 'a').close()
    
    elif os.path.exists(os.path.join(data_dir, 'task5')): 
        for i in range(F5_TESTS_NUM):
            x = np.load(os.path.join(data_dir, f'input_{i}', 'x.npy'))
            y = np.load(os.path.join(data_dir, f'input_{i}', 'y.npy'))
            res = functions.rle_scalar(x.tolist(), y.tolist())
            data = {'res': res}
    
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'wb') as f:
                pickle.dump(data, f)
        
        open(os.path.join(output_dir, 'task5'), 'a').close()

    elif os.path.exists(os.path.join(data_dir, 'task6_vectorised')): 
        for i in range(F6_TESTS_NUM):
            x = np.load(os.path.join(data_dir, f'input_{i}', 'x.npy'))
            y = np.load(os.path.join(data_dir, f'input_{i}', 'y.npy'))
            tick = time.time()
            res = functions_vectorised.cosine_distance(x, y)
            res_time = time.time() - tick
            data = {'res': res, 'time': res_time}
    
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'wb') as f:
                pickle.dump(data, f)
        
        open(os.path.join(output_dir, 'task6_vectorised'), 'a').close()

    elif os.path.exists(os.path.join(data_dir, 'task6')): 
        for i in range(F6_TESTS_NUM):
            x = np.load(os.path.join(data_dir, f'input_{i}', 'x.npy'))
            y = np.load(os.path.join(data_dir, f'input_{i}', 'y.npy'))
            res = functions.cosine_distance(x.tolist(), y.tolist())
            data = {'res': res}
    
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'wb') as f:
                pickle.dump(data, f)
        
        open(os.path.join(output_dir, 'task6'), 'a').close()


def check_test(data_dir):
    tol = 10
    output_dir = os.path.join(data_dir, 'output')
    gt_dir = os.path.join(data_dir, 'gt')
    
    if os.path.exists(os.path.join(output_dir, 'task1_vectorised')):
        for i in range(F1_TESTS_NUM):
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'rb') as f:
                data = pickle.load(f)

            with open(os.path.join(gt_dir, f'output_{i}.pkl') , 'rb') as f:
                gt_data = pickle.load(f)

            # check type
            if type(data['res']) not in (int, np.integer, np.int32, np.int64):
                res = f'functions_vectorised.py, sum_non_neg_diag: Wrong answer type'
                print(res)               
                return res
            
            # check answer
            if data['res'] != gt_data['res']:
                res = f'functions_vectorised.py, sum_non_neg_diag: Wrong answer on test {i}'
                print(res)               
                return res

            # check time
            if i == F1_TESTS_NUM - 1:
                if data['time'] > tol * gt_data['time']:
                    res = f"functions_vectorised.py, sum_non_neg_diag: Your solution is too slow! Execution time on test {i} is {data['time']:.4f} sec"
                    print(res)
                    return res  
        
        res = 'Ok'
        print('functions_vectorised.py, sum_non_neg_diag: Ok')   

    elif os.path.exists(os.path.join(output_dir, 'task1')):
        for i in range(F1_TESTS_NUM):
            with open(os.path.join(output_dir, f'output_{i}.pkl') , 'rb') as f:
                data = pickle.load(f)

            with open(os.path.join(gt_dir, f'output_{i}.pkl') , 'rb') as f:
                gt_data = pickle.load(f)
            
            # check type
            if type(data['res']) not in (int, np.integer, np.int32, np.int64):
                res = f'functions.py, sum_non_neg_diag: Wrong answer type'
                print(res)               
                return res

            # check answer
            if data['res'] != gt_data['res']:
                res = f'functions.py, sum_non_neg_diag: Wrong answer on test {i}'
                print(res)               
                return res
        
        res = 'Ok'
        print('functions.py, sum_non_neg_diag: Ok')   

    elif os.path.exists(os.path.join(output_dir, 'task2_vectorised')):
        for i in range(F2_TESTS_NUM):
            with open(os.path.join(output_dir, f'output_{i}.pkl'), 'rb') as f:
                data = pickle.load(f)

            with open(os.path.join(gt_dir, f'output_{i}.pkl'), 'rb') as f:
                gt_data = pickle.load(f)

            # check type
            if type(data['res']) not in (bool, np.bool_):
                res = f'functions_vectorised.py, are_multisets_equal: Wrong answer type'
                print(res)               
                return res

            # check answer
            if data['res'] != gt_data['res']:
                res = f'functions_vectorised.py, are_multisets_equal: Wrong answer on test {i}'
                print(res)
                return res

            # check time
            if i == F2_TESTS_NUM - 1:
                if data['time'] > tol * gt_data['time']:
                    res = f"functions_vectorised.py, are_multisets_equal: Your solution is too slow! Execution time on test {i} is {data['time']:.4f} sec"
                    print(res)
                    return res

        res = 'Ok'
        print('functions_vectorised.py, are_multisets_equal: Ok')  

    elif os.path.exists(os.path.join(output_dir, 'task2')):
        for i in range(F2_TESTS_NUM):
            with open(os.path.join(output_dir, f'output_{i}.pkl'), 'rb') as f:
                data = pickle.load(f)

            with open(os.path.join(gt_dir, f'output_{i}.pkl'), 'rb') as f:
                gt_data = pickle.load(f)

            # check type
            if type(data['res']) not in (bool, np.bool_):
                res = f'functions.py, are_multisets_equal: Wrong answer type'
                print(res)               
                return res

            # check answer
            if data['res'] != gt_data['res']:
                res = f'functions.py, are_multisets_equal: Wrong answer on test {i}'
                print(res)
                return res

        res = 'Ok'
        print('functions.py, are_multisets_equal: Ok')         

    elif os.path.exists(os.path.join(output_dir, 'task3_vectorised')):
        for i in range(F3_TESTS_NUM):
            with open(os.path.join(output_dir, f'output_{i}.pkl'), 'rb') as f:
                data = pickle.load(f)

            with open(os.path.join(gt_dir, f'output_{i}.pkl'), 'rb') as f:
                gt_data = pickle.load(f)

            # check type
            if type(data['res']) not in (int, np.integer, np.int32, np.int64):
                res = f'functions_vectorised.py, max_prod_mod_3: Wrong answer type'
                print(res)               
                return res

            # check answer
            if data['res'] != gt_data['res']:
                res = f'functions_vectorised.py, max_prod_mod_3: Wrong answer on test {i}'
                print(res)
                return res

            # check time
            if i == F3_TESTS_NUM - 1:
                if data['time'] > tol * gt_data['time']:
                    res = f"functions_vectorised.py, max_prod_mod_3: Your solution is too slow! Execution time on test {i} is {data['time']:.4f} sec"
                    print(res)
                    return res      

        res = 'Ok'
        print('functions_vectorised.py, max_prod_mod_3: Ok')

    elif os.path.exists(os.path.join(output_dir, 'task3')):
        for i in range(F3_TESTS_NUM):
            with open(os.path.join(output_dir, f'output_{i}.pkl'), 'rb') as f:
                data = pickle.load(f)

            with open(os.path.join(gt_dir, f'output_{i}.pkl'), 'rb') as f:
                gt_data = pickle.load(f)

            # check type
            if type(data['res']) not in (int, np.integer, np.int32, np.int64):
                res = f'functions.py, max_prod_mod_3: Wrong answer type'
                print(res)               
                return res

            # check answer
            if data['res'] != gt_data['res']:
                res = f'functions.py, max_prod_mod_3: Wrong answer on test {i}'
                print(res)
                return res
           

        res = 'Ok'
        print('functions.py, max_prod_mod_3: Ok')

    elif os.path.exists(os.path.join(output_dir, 'task4_vectorised')):
        for i in range(F4_TESTS_NUM):
            with open(os.path.join(output_dir, f'output_{i}.pkl'), 'rb') as f:
                data = pickle.load(f)

            with open(os.path.join(gt_dir, f'output_{i}.pkl'), 'rb') as f:
                gt_data = pickle.load(f)

            # check type
            if type(data['res']) != np.ndarray:
                res = f'functions_vectorised.py, convert_image: Wrong answer type'
                print(res)               
                return res

            # check answer
            if ((data['res'].shape != gt_data['res'].shape) or 
                (not np.allclose(data['res'], gt_data['res']))):
                res = f'functions_vectorised.py, convert_image: Wrong answer on test {i}'
                print(res)
                return res

            # check time
            if i == F4_TESTS_NUM - 1:
                if data['time'] > tol * gt_data['time']:
                    res = f"functions_vectorised.py, convert_image: Your solution is too slow! Execution time on test {i} is {data['time']:.4f} sec"
                    print(res)
                    return res      

        res = 'Ok'
        print('functions_vectorised.py, convert_image: Ok')   

    elif os.path.exists(os.path.join(output_dir, 'task4')):
        for i in range(F4_TESTS_NUM):
            with open(os.path.join(output_dir, f'output_{i}.pkl'), 'rb') as f:
                data = pickle.load(f)

            with open(os.path.join(gt_dir, f'output_{i}.pkl'), 'rb') as f:
                gt_data = pickle.load(f)

            # check type
            if type(data['res']) != list:
                res = f'functions.py, convert_image: Wrong answer type'
                print(res)               
                return res
            
            data['res'] = np.array(data['res'])
            gt_data['res'] = np.array(gt_data['res'])
            
            # check answer
            if ((data['res'].shape != gt_data['res'].shape) or 
                (not np.allclose(data['res'], gt_data['res']))):
                res = f'functions.py, convert_image: Wrong answer on test {i}'
                print(res)
                return res

        res = 'Ok'
        print('functions.py, convert_image: Ok')    

    elif os.path.exists(os.path.join(output_dir, 'task5_vectorised')):
        for i in range(F5_TESTS_NUM):
            with open(os.path.join(output_dir, f'output_{i}.pkl'), 'rb') as f:
                data = pickle.load(f)

            with open(os.path.join(gt_dir, f'output_{i}.pkl'), 'rb') as f:
                gt_data = pickle.load(f)


            # check type
            if type(data['res']) not in (int, np.integer, np.int32, np.int64):
                res = f'functions_vectorised.py, rle_scalar: Wrong answer type'
                print(res)               
                return res

            # check answer
            if data['res'] != gt_data['res']:
                res = f'functions_vectorised.py, rle_scalar: Wrong answer on test {i}'
                print(res)
                return res

            # check time
            if i == F5_TESTS_NUM - 1:
                if data['time'] > tol * gt_data['time']:
                    res = f"functions_vectorised.py, rle_scalar: Your solution is too slow! Execution time on test {i} is {data['time']:.4f} sec"
                    print(res)
                    return res      

        res = 'Ok'
        print('functions_vectorised.py, rle_scalar: Ok')   

    elif os.path.exists(os.path.join(output_dir, 'task5')):
        for i in range(F5_TESTS_NUM):
            with open(os.path.join(output_dir, f'output_{i}.pkl'), 'rb') as f:
                data = pickle.load(f)

            with open(os.path.join(gt_dir, f'output_{i}.pkl'), 'rb') as f:
                gt_data = pickle.load(f)


             # check type
            if type(data['res']) not in (int, np.integer, np.int32, np.int64):
                res = f'functions.py, rle_scalar: Wrong answer type'
                print(res)               
                return res

            # check answer
            if data['res'] != gt_data['res']:
                res = f'functions.py, rle_scalar: Wrong answer on test {i}'
                print(res)
                return res

        res = 'Ok'
        print('functions.py, rle_scalar: Ok')   

    elif os.path.exists(os.path.join(output_dir, 'task6_vectorised')):
        for i in range(F6_TESTS_NUM):
            with open(os.path.join(output_dir, f'output_{i}.pkl'), 'rb') as f:
                data = pickle.load(f)

            with open(os.path.join(gt_dir, f'output_{i}.pkl'), 'rb') as f:
                gt_data = pickle.load(f)

            # check type
            if type(data['res']) != np.ndarray:
                res = f'functions_vectorised.py, cosine_distance: Wrong answer type'
                print(res)               
                return res

            # check answer
            if ((data['res'].shape != gt_data['res'].shape) or 
                (not np.allclose(data['res'], gt_data['res']))):
                res = f'functions_vectorised.py, cosine_distance: Wrong answer on test {i}'
                print(res)
                return res

            # check time
            if i == F6_TESTS_NUM - 1:
                if data['time'] > tol * gt_data['time']:
                    res = f"functions_vectorised.py, cosine_distance: Your solution is too slow! Execution time on test {i} is {data['time']:.4f} sec"
                    print(res)
                    return res      

        res = 'Ok'
        print('functions_vectorised.py, cosine_distance: Ok')  

    elif os.path.exists(os.path.join(output_dir, 'task6')):
        for i in range(F6_TESTS_NUM):
            with open(os.path.join(output_dir, f'output_{i}.pkl'), 'rb') as f:
                data = pickle.load(f)

            with open(os.path.join(gt_dir, f'output_{i}.pkl'), 'rb') as f:
                gt_data = pickle.load(f)

            # check type
            if type(data['res']) != list:
                res = f'functions.py, cosine_distance: Wrong answer type'
                print(res)               
                return res

            data['res'] = np.array(data['res'])
            gt_data['res'] = np.array(gt_data['res'])
            
            # check answer
            if ((data['res'].shape != gt_data['res'].shape) or 
                (not np.allclose(data['res'], gt_data['res']))):
                res = f'functions.py, cosine_distance: Wrong answer on test {i}'
                print(res)
                return res 

        res = 'Ok'
        print('functions.py, cosine_distance: Ok') 

    return res


def grade(data_dir):
    results = load(open(os.path.join(data_dir, 'results.json')))
    max_mark = 27
    grade_mapping = [1.5] * 6 + [3] * 6 
    total_grade = 0
    ok_count = 0
    for result, grade in zip(results, grade_mapping):
        if 'Ok' in result['status']:
            total_grade += grade
            ok_count += 1
    total_count = len(results)
    description = '%02d/%02d' % (ok_count, total_count)
    mark = total_grade / sum(grade_mapping) * max_mark
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

        tests_dir = argv[1]

        results = []
        for input_dir in sorted(glob(os.path.join(tests_dir, '[0-9][0-9]_*_input'))):
            output_dir = sub('input$', 'check', input_dir)
            run_output_dir = os.path.join(output_dir, 'output')
            os.makedirs(run_output_dir, exist_ok=True)
            gt_src = sub('input$', 'gt', input_dir)
            gt_dst = os.path.join(output_dir, 'gt')
            if not os.path.exists(gt_dst):
                copytree(gt_src, gt_dst)

            try:
                start = time.time()
                run_single_test(input_dir, run_output_dir)
                end = time.time()
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

            test_num = os.path.basename(input_dir)[:2]
            if status == 'Runtime error' or status == 'Checker error':
                print(test_num, status, '\n', traceback)
                results.append({'status': status})
            else:
                results.append({'status': status})

        dump(results, open(os.path.join(tests_dir, 'results.json'), 'w'))
        res = grade(tests_dir)
        print('Mark:', res['mark'], res['description'])
