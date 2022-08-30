from tokenize import Double
import maxflow
from scipy import sparse
import numpy as np
import time
import json 

# THREADS = 32
ALGOS = ['parallel_AhujaOrlin_segment', 'parallel_push_relabel', 'parallel_push_relabel_segment']
A_SIZE = [ 100, 500, 1000, 5000, 10000, ] # 100, 500, 1000, 5000, 10000
THREADS = [10] #1, 4, 8, 12, 16, 20, 24, 28, 32


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
    result['Ave_load_time'] = dict(zip(alg_names, list(avg_maxflow_times[:,0])))
    result['Ave_solve_time']= dict(zip(alg_names, list(avg_maxflow_times[:,1])))
    return result


def run_tests():

    threads_result = {}
    for t in THREADS:
        threads_result[t] = bench_Solvers(10000,3, dense=True, matrix_generator=gen_matrix1, t=t)
    
    with open("threads_result.json", "w") as outfile:
        json.dump(threads_result, outfile, indent=2)

    # nodes_result = {}
    # for n in A_SIZE:
    #     nodes_result[n] = bench_Solvers(n,5, dense=True, matrix_generator=gen_matrix1, t=16)
    
    # with open("nodes_result.json", "w") as outfile:
    #     json.dump(nodes_result, outfile, indent=2)
    

    
run_tests()

# m = gen_matrix1(2,True)
# print(m)