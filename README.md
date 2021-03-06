# Final Project for Math 490

This repository contains all of the relevant files for Math 490's final project: a self-guided implementation of Graph Neural Networks to classify k-centers of a weighted, undirected graph.  (Completed Spring 2020)

### Abstract

In this project we implement two algorithms for solving the k-center problem on graph structured data, an exact algorithm based on minimum dominating sets, and a greedy approximation algorithm. We then transition to a Graph Neural Network approach to this problem. To do so, we train a graph attention network first proposed in *Graph Attention Networks* by Petar Velickovic, et. al on Erdos-Renyi graphs with random edge weights to discern k-centers and k-clusters. In doing so, we develop a new label-isomorphic approach to generating feature vectors that encodes intuitive structure of a given graph. Finally, we conclude with an evaluation of our efforts and future applications of such a GNN.

### License

[MIT](https://choosealicense.com/licenses/mit/)
