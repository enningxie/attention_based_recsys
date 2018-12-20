from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(processes=num_thread) as p:
        p.map(eval_one_rating, range(len(_testRatings)))