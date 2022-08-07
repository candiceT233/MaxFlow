import maxflow
from scipy import sparse
import numpy as np
import time


THREADS = 32
ALGOS = ['parallel_AhujaOrlin_segment', 'parallel_push_relabel', 'parallel_push_relabel_segment']
A_SIZE = [ 10000, 8000, 4000, 2000, 1000 , 100 ]

def gen_matrix1(n, density):
    return (sparse.rand(n,n,density=density,format='csr',random_state=42)*100).astype(np.uint32)
    # return (sparse.rand(n,n,density=density,format='csr')*100).astype(np.uint32)

def time_solvers(solver, alg_names, A, source, sink):
    times = np.zeros((len(alg_names), 2))
    for i, name in enumerate(alg_names):
        t1 = time.perf_counter()
        solver.load_graph(A, name)
        t2 = time.perf_counter()
        solver.solve(name, source, sink, THREADS)  
        t3 = time.perf_counter()
        times[i, 0] = t2 - t1       
        times[i, 1] = t3 - t2
    return times

def bench_Solvers(n, iters, seed=0, density=0.5, dense=True, matrix_generator=gen_matrix1):
    print("------------Running bench!--------------")
    print("n={}, iters={}, density={}, dense={}".format(n, iters, density, dense))
    np.random.seed(seed)

    S = maxflow.Solver()
    alg_names = ALGOS

    maxflow_times = np.zeros((len(alg_names), 2), dtype=np.float64)
    scipy_time = 0

    for i in range(iters):
        x = matrix_generator(n, density)
        if dense:
            x_ = x.toarray()
        else:
            x_ = x
        t1 = time.perf_counter()
        flow = sparse.csgraph.maximum_flow(x, 0, n-1).flow_value
        scipy_time += (time.perf_counter() - t1)
        maxflow_times += time_solvers(S, alg_names, x_, 0, n-1)

    print("Average time taken for scipy:", scipy_time/iters)
    avg_maxflow_times = maxflow_times/iters
    print("Average load time taken for maxflow algs:", list(zip(alg_names, list(avg_maxflow_times[:,0]))))
    print("Average solving time taken for maxflow algs:", list(zip(alg_names, list(avg_maxflow_times[:,1]))))

def run_tests():
    for n in A_SIZE:
        bench_Solvers(n,3, density=1, dense=True, matrix_generator=gen_matrix1)


run_tests()