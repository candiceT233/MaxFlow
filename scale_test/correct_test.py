from tokenize import Double
import maxflow
from scipy import sparse
import numpy as np
import time
import json 
import pandas as pd

# THREADS = 32
ALGOS = ['parallel_AhujaOrlin_segment', 'parallel_push_relabel', 'parallel_push_relabel_segment']



M1 = [[0,1,2,0,0,0], 
     [0,0,0,1,1,0],
     [0,0,0,5,5,0], 
     [0,0,0,0,0,5], 
     [0,0,0,0,0,5], 
     [0,0,0,0,0,0]]

M2 = [[0,0,2,0,0,0], 
     [0,0,0,0,0,0],
     [0,0,0,5,5,0], 
     [0,0,0,0,0,5], 
     [0,0,0,0,0,5], 
     [0,0,0,0,0,0]]


def gen_matrix1(matrix):
    # return (sparse.rand(n,n,density=density,format='csr',random_state=42)*100).astype(np.uint32)
    # m = np.random.randint(1, high=100, size=(n, n), dtype=np.uint32)
    # matrix[matrix == 0] = -1 # negative not work?
    m = np.matrix(matrix)
    # m = np.full((n, n), 0)
    # m[2,1] = 1

    print(f"Input Matrix = \n{m}\n")
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
        print(f"{name} Max Flow Value = {flow2}")
    return times

def bench_Solvers(n, iters, matrix, seed=0, dense=True, t=10):
    print("------------Running bench!--------------")
    matrix = np.array(matrix)
    edges = len(matrix[np.nonzero(matrix)])
    
    print("vertices={}, edges={}, threads={}, iters={}, dense={}".format(n, edges, t, iters, dense))
    np.random.seed(seed)

    S = maxflow.Solver()
    alg_names = ALGOS

    maxflow_times = np.zeros((len(alg_names), 2), dtype=np.float64)
    scipy_time = 0

    for i in range(iters):
        x = gen_matrix1(matrix)
        print(f"sparse.csr_matrix Edges and Capacity = \n{x}\n")
        if dense:
            x_ = x.toarray()
        else:
            x_ = x
        t1 = time.perf_counter()
        flow = sparse.csgraph.maximum_flow(x, 0, n-1).flow_value
        print(f"sparse.csgraph.maximum_flow = {flow}")
        scipy_time += (time.perf_counter() - t1)
        maxflow_times += time_solvers(S, alg_names, x_, 0, n-1, flow, t) # modify source and sink for graph
        
    avg_maxflow_times = maxflow_times/iters

    # print("Ave_graph_gen_time:", scipy_time/iters)
    # print("Ave_load_time:", list(zip(alg_names, list(avg_maxflow_times[:,0]))))
    # print("Ave_solve_time:", list(zip(alg_names, list(avg_maxflow_times[:,1]))))

    result= {}
    result['Edges'] = edges
    result['Ave_graph_read_time'] = scipy_time/iters
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

def run_tests(m):
    
    fix_size=len(m)
    thread=1

    best_threads_result = {}
    
    best_threads_result[thread] = bench_Solvers(fix_size,1, m, dense=False, t=thread)

    df = pd.DataFrame.from_dict(best_threads_result,orient='columns')

    print(df)


    
run_tests(M1)
run_tests(M2)

# m = gen_matrix1(2,True)
# print(m)