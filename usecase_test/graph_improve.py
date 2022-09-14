from tokenize import Double
import maxflow
from scipy import sparse
import numpy as np
import time
import json 
import pandas as pd
import heapq

# THREADS = 32
ALGOS = ['parallel_AhujaOrlin_segment', 'parallel_push_relabel', 'parallel_push_relabel_segment']
columns = ['Edge','Threads','Load-parallel_AhujaOrlin_segment',
'Solve-parallel_AhujaOrlin_segmente_',
'Load-parallel_AhujaOrlin_segment',
'Solve-parallel_push_relabel',
'Load-parallel_push_relabel_segment',
'Solve-parallel_push_relabel_segment',]
MAX_VAL = 100
LOW_VAL = 50

def create_matrix():
    m = np.zeros((22, 22)).astype(int)
    # initialize all score as 5 as best performance (low bottleneck)
    m[0,1:5] = MAX_VAL
    for i in range(1,17):
        m[i,i+4] = MAX_VAL
    m[17:21,21] = MAX_VAL

    # populate some edges wtih 1 as slow performance (high bottleneck)
    low_val = MAX_VAL/2
    m[1,5] = low_val
    m[4,8] = low_val
    m[6,10] = low_val
    m[9,13] = low_val
    m[11,15] = low_val
    m[15,19] = low_val
    m[18,21] = low_val

    # print(m)
    return m


def time_solvers(solver, alg_names, A, source, sink, flow, t=10):
    times = np.zeros((len(alg_names), 2))
    finalflow = 0
    for i, name in enumerate(alg_names):
        t1 = time.perf_counter()
        solver.load_graph(A, name)
        t2 = time.perf_counter()
        flow2 = solver.solve(name, source, sink, t)  
        t3 = time.perf_counter()
        times[i, 0] = t2 - t1       
        times[i, 1] = t3 - t2
        assert flow == flow2, "Error at iteration:{}! : function {} gives flow:{}, while scipy's maxflow:{}".format(i, name, flow2, flow)
        # print(f"{name} Max Flow Value = {flow2}")
        finalflow = flow2 # only store 1 value is fine
    return times, finalflow

def bench_Solvers(n, matrix, seed=0, dense=True, t=10):
    matrix = np.array(matrix)
    edges = matrix.shape[0]
    # print("------------Running bench!--------------")
    # print("vertices={}, edges={}, threads={}, dense={}".format(n, edges, t, dense))

    S = maxflow.Solver()
    alg_names = ALGOS

    maxflow_times = np.zeros((len(alg_names), 2), dtype=np.float64)
    scipy_time = 0
    maxflow_times = 0
    finalflow = 0
    
    x = np.copy(matrix)
    x = sparse.csr_matrix(x)
    # print(f"sparse.csr_matrix Edges and Capacity = \n{x}\n")
    if dense:
        x_ = x.toarray()
    else:
        x_ = x
    t1 = time.perf_counter()
    flow = sparse.csgraph.maximum_flow(x, 0, n-1).flow_value
    # print(f"sparse.csgraph.maximum_flow = {flow}")
    scipy_time += (time.perf_counter() - t1)
    times, finalflow = time_solvers(S, alg_names, x_, 0, n-1, flow, t) # modify source and sink for graph
    maxflow_times += times
        
    avg_maxflow_times = maxflow_times

    result= {}
    result['max_flow'] = finalflow
    result['Edges'] = edges
    result['Ave_graph_read_time'] = scipy_time
    l_alg_names = [f"Load-{x}" for x in alg_names]
    s_alg_names = [f"Solve-{x}" for x in alg_names]

    result.update(dict(zip(l_alg_names, list(avg_maxflow_times[:,0]))))  
    result.update(dict(zip(s_alg_names, list(avg_maxflow_times[:,1]))))
    return result

def apply_opt(arr,edges,speedup=1.45):
    t1 = time.perf_counter()
    for val in edges:
        # improves 45%, no longer improves when reaching maximum
        if (val * speedup) <= MAX_VAL: 
            # improve only the 1st found edge with 
            arr[(arr == val).nonzero()[0][:1]] = val * speedup
        else:
            arr[(arr == val).nonzero()[0][:1]] = MAX_VAL
    t2 = time.perf_counter()
    print(f"apply_opt time (sec): {t2 - t1} ")
    return arr

def improve_slowest(matrix, pc=0.1):
    size = matrix.shape[0]
    num = int(size*pc)

    t1 = time.perf_counter()
    x = matrix.ravel()
    # x[x[:,9].argsort()]

    # find nth smallest edges that's not zero
    edges = heapq.nsmallest(num,x[x != 0]) 
    t2 = time.perf_counter()
    print(f"find_slowest time (sec): {t2 - t1}")
    print(f"The {pc*100}% slowest edges are: {edges}")

    x = apply_opt(x,edges)

    # convert back into matrix
    return np.reshape(x, (size, size))


def improvement(prev_flow,curr_flow):
    return ((curr_flow - prev_flow)/prev_flow)

def max_flow(matrix):
    
    fix_size=matrix.shape[0]

    thread=1
    # best_threads_result = {}
    # best_threads_result[thread] = bench_Solvers(fix_size,1, matrix, dense=False, t=thread)
    # df = pd.DataFrame.from_dict(best_threads_result,orient='index')

    return bench_Solvers(fix_size, matrix, dense=False, t=thread)

if __name__ == "__main__":

    threshold = 0.05 # algorithm stops when improvement is lower than 5%
    print(f"Algorithm stops with improvement less than {threshold*100}%\n")
    
    t1 = time.perf_counter()
    matrix = create_matrix()
    print(f"Initial edge values are {sparse.csr_matrix(matrix)}\n")
    result = max_flow(matrix)
    # print(result)

    base_flow = result['max_flow']
    print(f"Baseline maxflow is {base_flow}")
    prev_flow = base_flow
    curr_flow = base_flow # set curr_flow large to enter while loop
    iterate = 0 # for counting iterations
    imp = 10

    while imp > threshold:
        prev_flow = curr_flow
        matrix = improve_slowest(matrix)
        result = max_flow(matrix)
        curr_flow = result['max_flow']
        print(f"New maxflow is {curr_flow}")
        imp = improvement(prev_flow,curr_flow)
        print(f"Flow improvement is {imp*100}%\n")
        iterate +=1
    
    t2 = time.perf_counter()

    print(f"----------------- Final Output -----------------")
    print(f"Program total time (sec): {t2 - t1}")
    print(f"Iteration count is {iterate}")
    print(f"Final flow is {curr_flow}")
    
    print(f"Final edge values are {sparse.csr_matrix(matrix)}\n")
    # print(sparse.csr_matrix(matrix))
    

