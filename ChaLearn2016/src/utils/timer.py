'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors : now more
Description : 
LastEditTime: 2020-10-09 08:14:53
'''
import time
from functools import wraps
from inspect import signature
from contextlib import contextmanager

def display_result(print_result=True,print_type=True):
	def decorator(func):
		if not __debug__:
			return func

		@wraps(func)
		def wrapper(*args,**kwargs):
			result = func(*args,**kwargs)
			if print_result and print_type:
				print(func.__module__,func.__name__,type(result),result)
			elif print_result and not(print_type):
				print(func.__module__,func.__name__,result)
			elif not(print_result) and print_type:
				print(func.__module__,func.__name__,type(result))
			return result
		return wrapper
	return decorator


@contextmanager
def time_code_block(label):
	start = time.process_time()
	try:
		yield
	finally:
		end = time.process_time()
		print(label,end-start)

		
'''
@description: Decorator that report the execution time
@param      : Function
@return: 	: None
'''
def timethis(func):
	''' Decorator that report the execution time '''
	if not __debug__:
		return func
		
	@wraps(func)
	def wrapper(*args,**kwargs):
		start = time.process_time()
		resault = func(*args,**kwargs)
		end = time.process_time()
		print(func.__module__,func.__name__,end - start)
		return resault
	return wrapper



'''
@description  : 对函数参数做类型检查 
@param {type} : 指定参数类型
@return : None
'''
def typeassert(*ty_args,**ty_kwargs):
	def decorate(func):
		# If not in optimized mode,disabled type checking
		if not __debug__:
			return func

		# Map function argument names to supplied types
		sig = signature(func)
		bound_types = sig.bind_partial(*ty_args,**ty_kwargs).arguments

		@wraps(func)
		def wrapper(*args,**kwargs):
			bound_values = sig.bind(*args,**kwargs)
			# Enforce type assertions across supplied arguments 
			for name,value in bound_values.arguments.items():
				if name in bound_types:
					if not isinstance(value,bound_types[name]):
						raise TypeError('Arguments {0} must be {1}'.format(name,bound_types[name]))
			return func(*args,**kwargs)
		return wrapper
	return decorate