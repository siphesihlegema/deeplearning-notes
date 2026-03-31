# Matrix multiplication

Let $A$ be an $m \times n$ matrix and *B* be an $n \times p$ matrix. Their product $C = AB$ is an $m \times p$ matrix where each entry is:
$$
C_{ij} = \sum_{k=1}^{n}A_{ik}B_{kj}
$$
The entry in row $i$, column of $C$ is the **dot product** of row $i$ of $A$ with column $j$ of $B$

*Example:*

Let
$$
A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad
B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}
$$
Then $C = AB$:
$$
C_{11} = (1)(5) + (2)(7) = 19, \quad C_{12} = (1)(6) + (2)(8) = 22
$$
$$
C_{21} = (3)(5) + (4)(7) = 43, \quad C_{22} = (3)(6) + (4)(8) = 50
$$
$$
\therefore \; AB = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}
$$
