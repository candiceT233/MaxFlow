from tokenize import Double
import maxflow
from scipy import sparse
import numpy as np
import time
import json 
import pandas as pd

# THREADS = 32
ALGOS = ['parallel_AhujaOrlin_segment', 'parallel_push_relabel', 'parallel_push_relabel_segment']
A_SIZE = [ 10, 100, 1000, 10000 ] # weak scaling, x4
THREADS = [1,2,4,8,16,32,64,128] # strong scaling, x2


def gen_matrix1(n):
    # return (sparse.rand(n,n,density=density,format='csr',random_state=42)*100).astype(np.uint32)
    m = np.random.randint(1, high=100, size=(n, n), dtype=np.uint32)
    return (sparse.csr_matrix(m))
    # return (sparse.rand(n,n,density=density,format='csr')*100).astype(np.uint32)

def time_solvers(solver, alg_names, A, source, sink, flow, t=10):
    times = np.zeros((len(alg_names), 2))
    for i, name in enumerate(alg_names):
        t1 = time.perf_counter()
        solver.load_graph(A, name)
        t2 = time.perf_counter()
        flow2 = solver.solve(name, source, sink, t)  
        t3 = time.perf_counter()
        times[i, 0] = t2 - t1       
        times[i, 1] = t3 - t2
        assert flow == flow2, "Error at iteration:{}! : function {} gives flow:{}, while scipy's maxflow:{}".format(i, name, flow2, flow)
    return times

def bench_Solvers(n, iters, seed=0, dense=True, matrix_generator=gen_matrix1, t=10):
    print("------------Running bench!--------------")
    edges = int(n * (n -1)/2) 
    print("vertices={}, edges={}, threads={}, iters={}, dense={}".format(n, edges, t, iters, dense))
    np.random.seed(seed)

    S = maxflow.Solver()
    alg_names = ALGOS

    maxflow_times = np.zeros((len(alg_names), 2), dtype=np.float64)
    scipy_time = 0

    for i in range(iters):
        x = matrix_generator(n)
        if dense:
            x_ = x.toarray()
        else:
            x_ = x
        t1 = time.perf_counter()
        flow = sparse.csgraph.maximum_flow(x, 0, n-1).flow_value
        scipy_time += (time.perf_counter() - t1)
        maxflow_times += time_solvers(S, alg_names, x_, 0, n-1, flow, t)
        

    print("Ave_graph_gen_time:", scipy_time/iters)
    avg_maxflow_times = maxflow_times/iters
    print("Ave_load_time:", list(zip(alg_names, list(avg_maxflow_times[:,0]))))
    print("Ave_solve_time:", list(zip(alg_names, list(avg_maxflow_times[:,1]))))
    result= {}
    result['Edges'] = edges
    result['Ave_graph_gen_time'] = scipy_time/iters
    l_alg_names = [f"Load-{x}" for x in alg_names]
    s_alg_names = [f"Solve-{x}" for x in alg_names]

    result.update(dict(zip(l_alg_names, list(avg_maxflow_times[:,0]))))  
    result.update(dict(zip(s_alg_names, list(avg_maxflow_times[:,1]))))
    return result

columns = ['Edge','Threads','Load-parallel_AhujaOrlin_segment',
'Solve-parallel_AhujaOrlin_segmente_',
'Load-parallel_AhujaOrlin_segment',
'Solve-parallel_push_relabel',
'Load-parallel_push_relabel_segment',
'Solve-parallel_push_relabel_segment',]

def run_tests():

    # threads_result = {}
    # for t in THREADS:
    #     threads_result[t] = bench_Solvers(2560,5, dense=True, matrix_generator=gen_matrix1, t=t)

    # df = pd.DataFrame.from_dict(threads_result,orient='columns')
    # df.to_csv('scale_threads_result.csv', index=True)
    
    fix_size=5000
    best_threads_result = {}
    for t in range(1,65):
        best_threads_result[t] = bench_Solvers(fix_size,5, dense=True, matrix_generator=gen_matrix1, t=t)
    
    best_threads_result[128] = bench_Solvers(fix_size,5, dense=True, matrix_generator=gen_matrix1, t=128)

    df = pd.DataFrame.from_dict(best_threads_result,orient='columns')
    df.to_csv('best_threads_result.csv', index=True)

    # for t in THREADS:

    #     nodes_result = {}
    #     for n in A_SIZE:
    #         nodes_result[n] = bench_Solvers(n,5, dense=True, matrix_generator=gen_matrix1, t=t)

    #     df = pd.DataFrame.from_dict(nodes_result,orient='index')
    #     df.to_csv(f't_{t}_scale_nodes_result.csv', index=True)
    

    
run_tests()

# m = gen_matrix1(2,True)
# print(m)