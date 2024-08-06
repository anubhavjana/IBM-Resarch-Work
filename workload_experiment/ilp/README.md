<h1> Integer Linear Programming to optimize instance configuration </h1>

## Notation

- **Gi,j**: Binary variables indicating GPU i operated in mode j, \(1 \leq i \leq N\), \(1 \leq j \leq R\).
- **Ai,j**: Aggregate rate for GPU i in mode j.
- **U1, U2, ..., UM**: M user classes with:
  - Request (query) rates \(Q1, Q2, ..., QM\).
  - Latency bounds \(L1, L2, ..., LM\).
- **Xu,i,j**: Decision variables denoting the rate allocation to user \(Uu\) on GPU i in mode j.
- **fi,j(r)**: Denotes the latency in GPU i mode j for rate r.

## Model Formulation

Minimize

\[
\sum_{i=1}^{N} \sum_{j=1}^{R} G_{i,j}
\]

Subject to:

\[
\sum_{j=1}^{R} G_{i,j} \leq 1, \forall i
\]

\[
\sum_{i,j} X_{u,i,j} = Q_u, \forall u
\]

\[
\sum_{u,j} X_{u,i,j} \leq \sum_{j} G_{i,j} \cdot A_{i,j}, \forall i
\]

\[
f_{i,j} \left( \sum_{k=1}^{u} X_{k,i,j} \right) \leq L_u, \forall i, j, u
\]

\[
G_{i,j} \in \{0, 1\}, \forall i, j
\]

## How to run ## 

> python3 anubhav_ilp.py -- will generate the user configuration to be fed to router 
