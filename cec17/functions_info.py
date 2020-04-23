"""Detailed data dictionaries for basic functions from CEC2017
"""
__author__ = "Kostrzewa Lukasz, Napieralski Adam"

available_functions = [
    'bent_cigar',
    'zakharov',
    'rosenbrock',
    'rastrigin',
    'expanded_shaffer_f6',
    'levy',
    'schwefel'
]

F_min = {
    'bent_cigar': 100,
    'zakharov': 200,
    'rosenbrock': 300,
    'rastrigin': 400,
    'expanded_shaffer_f6': 500,
    'levy': 800,
    'schwefel': 900
}

function_number = {
    'bent_cigar': 1,
    'zakharov': 3,
    'rosenbrock': 4,
    'rastrigin': 5,
    'expanded_shaffer_f6': 6,
    'levy': 9,
    'schwefel': 10
}