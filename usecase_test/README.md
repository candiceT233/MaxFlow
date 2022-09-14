# Example Algorithm Process in Evaluating Bottleneck Score (simplified_HACC):
The example simplified HACC workflow graph: https://docs.google.com/drawings/d/1cb3N4AH-CPtHd-5Okt7XsRwDJsHuFmwc5zLZHDJA8A4/edit
### Definitions
- High score as small bottleneck, low score as a large bottleneck
0 All scores must be greater than 0, must be integer (for the algorithm to work)
### Steps:
1. Convert bottleneck score into inverse (finding max flow meaning finding the minimum bottleneck edges in the current graph
2. Find the baseline maxflow number of the graph
3. Sort the edges from lowest to highest (** can use parallel sort)
4. Improve the 10% lowest score ( simulate applying optimization )
5. Re-evaluate graph if maxflow improves
6. Repeat steps 3-5 until improvement is less than a threshold ( 1%? user defined?)
