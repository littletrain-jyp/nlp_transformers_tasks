# -*- coding: utf-8 -*-
# @Time         : 2023/5/23 19:36
# @Author       : Yupeng Ji
# @File         : print_run_time.py
# @Description  : 计算某个函数的运行时间

import time


def print_run_time(func):
    """时间装饰器"""

    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        print(f'[file:{__file__}][{func.__name__}] run time is {time.time() - local_time :4f}')

    return wrapper


@print_run_time
def test():
    time.sleep(5)
    print('test运行完毕')


if __name__ == '__main__':
    test()