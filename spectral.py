#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#%%
"""
1. K5: 5-clique, or a fully connected simple, undirected graph of 5 nodes
2. K5.3: two disconnected components, C1 and C2, where C1 is a 5-clique K5 and C2 is a 3-clique K3
3. K5.3e: Almost the same as K53 but there is a single edge connecting the two components
4. B2.3: Complete bi-partite graph with n1=2 nodes in the first part and n2=3 nodes in the second part
5. S5: A 5-vertex star(one central "hub" node that connects to all other "spoke" nodes)
6. P5: A simple path of 5 vertices
"""
# K5: 5-clique, or a fully connected simple, undirected graph of 5 nodes
K5 = nx.complete_graph(5)
adjacencyMatrixK5 = nx.adjacency_matrix(K5)
print('Adjacency matrix for K5')
print(adjacencyMatrixK5.todense())

# K5.3: two disconnected components, c1 and c2, where C1 is a 5-clique K5 and C2 is a 3-clique K3
K3 = nx.complete_graph(3)
K53 = nx.disjoint_union(K5,K3)
adjacencyMatrix53 = nx.adjacency_matrix(K53)
print('Adjacency matrix for K5.3')
print(adjacencyMatrix53.todense())

# K5.3e: Almost the same as K53 but there is a single edge connecting the two components
K53e = nx.disjoint_union(K5,K3)
K53e.add_edge(5,1)
adjacencyMatrixK53e = nx.adjacency_matrix(K53e)
print('Adjacency matrix for K5.3e')
print(adjacencyMatrixK53e.todense())

# B2.3: Complete bi-partite graph with n1=2 nodes in the first part and n2=3 nodes in the second part
B23 = nx.complete_multipartite_graph(2,3)
adjacencyMatrixB23 = nx.adjacency_matrix(B23)
print('Adjacency matrix for B2.3')
print(adjacencyMatrixB23.todense())

# S5: A vertex star (one central "hub" node that connects to all the other "spoke" nodes)
S5 = nx.star_graph(4)
adjacencyMatrixS5 = nx.adjacency_matrix(S5)
print('Adjacency matrix for S5')
print(adjacencyMatrixS5.todense())

# P5: A simple path of 5 vertices
P5=nx.path_graph(5)
adjacencyMatrixP5 = nx.adjacency_matrix(P5)
print('Adjacency matrix for P5')
print(adjacencyMatrixP5.todense())

#%%
"""
Generating the degree matrix D for a given adjacency matrix A.
 Write a script that generates a degree matrix for each of the adjacency 
 matrices. Note that the degree matrix is a diagonal matrix 
 where all the positions except for the diagonal are zero's. 
 The diagonal elements correspond to the degrees of the corresponding nodes, 
 namely Dii = degree(v_i).
"""
#%%
# Degree Matrix for K5
i = K5.number_of_nodes()
d = np.zeros((i,i),dtype=int)
i = 0
for node in K5.nodes():
    d[i,i] = K5.degree(node)
    i = i + 1
print('Degree Matrix for K5')
print(d)
    
# Degree Matrix for K5.3
i = K53.number_of_nodes()
d = np.zeros((i,i),dtype=int)
i = 0
for node in K53.nodes():
    d[i,i] = K53.degree(node)
    i = i + 1
print('Degree Matrix for K5.3')
print(d)

# Degree Matrix for K5.3e
i = K53e.number_of_nodes()
d = np.zeros((i,i),dtype=int)
i = 0
for node in K53e.nodes():
    d[i,i] = K53e.degree(node)
    i = i + 1
print('Degree Matrix for K5.3e')
print(d)

# Degree Matrix for B2.3
i = B23.number_of_nodes()
d = np.zeros((i,i),dtype=int)
i = 0
for node in B23.nodes():
    d[i,i] = B23.degree(node)
    i = i + 1
print('Degree Matrix for B2.3')
print(d)

# Degree Matrix for S5
i = S5.number_of_nodes()
d = np.zeros((i,i),dtype=int)
i = 0
for node in S5.nodes():
    d[i,i] = S5.degree(node)
    i = i + 1
print('Degree Matrix for S5')
print(d)

# Degree Matrix for P5
i = P5.number_of_nodes()
d = np.zeros((i,i),dtype=int)
i = 0
for node in P5.nodes():
    d[i,i] = P5.degree(node)
    i = i + 1
print('Degree Matrix for P5')
print(d)



#%%
"""
Generating the graph Laplacian matrix L for a given adjacency matrix A and its degree matrix D.
Write a script that generates the graph Laplacian matrix L = D - A for each of the 
adjacency matrices 
"""
L = nx.laplacian_matrix(K5)
print('Laplacian matrix for K5')
print(L.todense())

L = nx.laplacian_matrix(K53)
print('Laplacian matrix for K5.3')
print(L.todense())

L = nx.laplacian_matrix(K53e)
print('Laplacian matrix for K5.3e')
print(L.todense())

L = nx.laplacian_matrix(B23)
print('Laplacian matrix for B2.3')
print(L.todense())

L = nx.laplacian_matrix(S5)
print('Laplacian matrix for S5')
print(L.todense())

L = nx.laplacian_matrix(P5)
print('Laplacian matrix for P5')
print(L.todense())

#%%
"""
Is L a sparse matrix?
In what positions does L have non-zero elements?
What are the values of the non-diagonal and non-zero elements?
What does L contain along its diagonal?
Answers:

No, the Laplacian matrix is not a sparse matrix.
L has non-zero elements on the diagonal(contains degree of the node) and the cells where an edge is present between the 2 nodes(non-empty in adjacency matrix).
Non-diagonal and non-zero elements have the value -1.
Along the diagonal, L contains the value of degrees of the corresponding nodes.
"""

"""
 Generating the graph spectrum, or the multiset of the eigenvalues of the graph adjacency matrix A. Write a script that calculates the eigenvalues of the 
 graph adjacency matrix.Plot the eigenvalues in the increasing order of their values
"""
ev = nx.adjacency_spectrum(K5)
print('Graph spectrum for adjacency matrix of K5')
ev = [ x.real for x in ev ]
print(ev)
plt.scatter(range(K5.number_of_nodes()), np.sort(ev))
plt.grid(True)
plt.show()

ev = nx.adjacency_spectrum(K53)
print('Graph spectrum for adjacency matrix of K5.3')
ev = [ x.real for x in ev ]
print(ev)
plt.scatter(range(K53.number_of_nodes()), np.sort(ev))
plt.grid(True)
plt.show()

ev = nx.adjacency_spectrum(K53e)
print('Graph spectrum for adjacency matrix ofK5.3e')
ev = [ x.real for x in ev ]
print(ev)
plt.scatter(range(K53e.number_of_nodes()), np.sort(ev))
plt.grid(True)
plt.show()

ev = nx.adjacency_spectrum(B23)
print('Graph spectrum for adjacency matrix of B2.3')
ev = [ x.real for x in ev ]
print(ev)
plt.scatter(range(B23.number_of_nodes()), np.sort(ev))
plt.grid(True)
plt.show()

ev = nx.adjacency_spectrum(S5)
print('Graph spectrum for adjacency matrix of S5')
ev = [ x.real for x in ev ]
print(ev)
plt.scatter(range(S5.number_of_nodes()), np.sort(ev))
plt.grid(True)
plt.show()

ev = nx.adjacency_spectrum(P5)
print('Graph spectrum for adjacency matrix of P5')
ev = [ x.real for x in ev ]
print(ev)
plt.scatter(range(P5.number_of_nodes()), np.sort(ev))
plt.grid(True)
plt.show()


#%%
"""
1. What can you say about the eigenvalues of the complete graph(K5): The number of unique eigenvalues,
the largest and the smallest eigenvalues, the multiplicity (how many times the same eigenvalue appears)
of eigenvalue?
A. For the complete graph K5: There are only 2 unque eigenvalues -1 and (n-1) where n is the number of nodes in the graph.
The largest eigenvalue is n-1 i.e. 4 and smallest is -1. The largest eigenvalue appears only once and the
second eigenvalue -1 has multiplicity of n-1 or 4 in this case.
2. What is the graph spectrum of the bi-partite graph, B2.3? if n1=n and n2=m (a general complete bi-partite graph),
then what can you say about its  graph spectrum? [Hint: check sqrt(n*m)] if \lambda is the eigenvalue of the bi-partite graph, will minus \lambda be also the eigenvalue?
A. The graph spectrum for bi-partite graph B2.3 is [2.4494897427831788, -2.4494897427831774, 2.8866158951433349e-16, 9.6660067518829571e-48, 1.2154326714572542e-63] 
(ignoring the imaginary part) which is approximately [2.45, -2.45, 0, 0, 0]. For a bi-partite graph Bnm, two eigenvalues will be sqrt(nm) and -sqrt(nm) and the 
remaining eigenvalues are zero. Yes, if lambda is the eigenvalue of the bi-partite graph, minus lambda will also be the eigenvalue     
3. What is the largest eigenvalue of the star graph S5? If S5 were generalized to a N-vertex star, What could you say about the value of its eigenvalue?
A. The largest eigenvalue for the star graph S5 is 2. So the largest eigenvalue for N-vertex would be sqrt(N-1)
4. What is the largest eigenvalue of the path graph P5? As the length of the path increases, what can you say about the changes in the largest eigenvalue?
A. The largest eigenvalue of the path graph P5 is 1.7320508075688805. As, the length of the path increases, 
the largest eigenvalue becomes closer to 2.
5. How does the largest eigenvalue of the path P5 (or its more generalization to an arbitrary 
length) compare with the largest eigenvalues of the star graph or the complete graph? If you 
are asked to sort the largest eigenvalue of the path, the star, and the clique) in the increasing
order, what kind of relationship would you assign (E.g., \lambda{path} > or < than \lambda{star})?
A. In this case, the largest eigenvalue of P5(1.73) is lesser than both that of S5(2) and K5(4). In general,
lambda(path) <= lambda(star) <= lambda(clique)
"""

"""
Generating the graph spectrum, or the multiset of the eigenvalues of the graph Laplacian. Write a script that 
calculates the eigenvalues of the graph laplacian for each of the graphs
"""
ev = nx.laplacian_spectrum(K5)
print('Graph spectrum for graph Laplacian of K5')
ev = [ x.real for x in ev ]
print(ev)
plt.scatter(range(K5.number_of_nodes()), np.sort(ev))
plt.grid(True)
plt.show()

ev = nx.laplacian_spectrum(K53)
print('Graph spectrum for graph Laplacian of K5.3')
ev = [ x.real for x in ev ]
print(ev)
plt.scatter(range(K53.number_of_nodes()), np.sort(ev))
plt.grid(True)
plt.show()

ev = nx.laplacian_spectrum(K53e)
print('Graph spectrum for graph Laplacian ofK5.3e')
ev = [ x.real for x in ev ]
print(ev)
plt.scatter(range(K53e.number_of_nodes()), np.sort(ev))
plt.grid(True)
plt.show()

ev = nx.laplacian_spectrum(B23)
print('Graph spectrum for graph Laplacian of B2.3')
ev = [ x.real for x in ev ]
print(ev)
plt.scatter(range(B23.number_of_nodes()), np.sort(ev))
plt.grid(True)
plt.show()

ev = nx.laplacian_spectrum(S5)
print('Graph spectrum for graph Laplacian of S5')
ev = [ x.real for x in ev ]
print(ev)
plt.scatter(range(S5.number_of_nodes()), np.sort(ev))
plt.grid(True)
plt.show()

ev = nx.laplacian_spectrum(P5)
print('Graph spectrum for graph Laplacian of P5')
ev = [ x.real for x in ev ]
print(ev)
plt.scatter(range(P5.number_of_nodes()), np.sort(ev))
plt.grid(True)
plt.show()

#%%
"""
1. What can you say about the largest and the smallest eigenvalues?
A. The smallest eigenvalue for all the graphs is 0. The largest eigenvalue for the graphs is as follows:
   K5 5 Number of nodes
   K5.3 5 Number of nodes in the larger clique
   K5.3r 6.14
   B2.3 5 Number of nodes
   S5 5 Number of nodes P5 3.6

2. What is the multiplicity(how many times the same eigenvalue appears) of the zero eigenvalue for each of the cases?
A. Multiplicity of the zero eigenvalue - K5 1 K5.3 2 K5.3e 1 B2.3 1 S5 1 P5 1

3. If K53 graph would be generalized to include c>2 componenets, what can you say about the multiplicity
of the zero eigenvalues?
A. The multiplicity of the zero eigenvalue is equal to the number of the disconnected components in the graph

4. If graph G is connected (i.e., the number of disconnected components is one). what can you say about
the multiplicity of the zedro eigenvalue?
A. If the graph is connected, then the multiplicity of zero eigenvalue is 1.

5. For the bi-partite graph, what is the value of the second smallest eigenvalue?
A. For the bi-partite graph B2.3, the second smallest eigenvalue is 2 i.e. minimum os (2,3)/

6. Is the vector, whose components consist of 1's only, the eigenvector of the Laplacian? If it is, then what is
its corresponding eigenvalue?
A. Yes, the vector with all 1s is the eigenvector of the Laplacian. The corresponding eigenvalue is 0.

7. Suppose the graph Laplacian matrix has the zero eigenvalue of multiplicity k. Can you say anything about the 
structure of such a graph?
A. If the zero eigenvalue has multiplicity k, the graph has k disconnected components.