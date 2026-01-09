#!/usr/bin/env python3
'''docstring to create bar'''
import numpy as np
import matplotlib.pyplot as plt


def bars():
    '''to create bar(fruits)'''
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    categories = ['Farrah', 'Fred', 'Felicia']
    apples, bananas, oranges, peaches = fruit[0], fruit[1], fruit[2], fruit[3]
    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    ls1 = apples+bananas
    ls2 = apples+bananas+oranges
    plt.bar(categories, apples, color='red', width=0.5)
    plt.bar(categories, bananas, bottom=apples, color='yellow', width=0.5)
    plt.bar(categories, oranges, bottom=ls1, color='#ff8000', width=0.5)
    plt.bar(categories, peaches, bottom=ls2, color='#ffe5b4', width=0.5)
    plt.legend(['apples', 'bananas', 'oranges', 'peaches'])
