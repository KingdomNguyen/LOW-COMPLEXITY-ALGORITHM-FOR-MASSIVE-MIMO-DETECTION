# LOW-COMPLEXITY-ALGORITHM-FOR-MASSIVE-MIMO-DETECTION

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

*High efficient algorithm for massive mimo detection*

## Table of Contents
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Installation](#installation)
- [Results](#results)
- [References](#references)
## Problem Statement

The problem considered is MIMO (Multiple-Input Multiple-Output) detection in a wireless communication system using 16-QAM (Quadrature Amplitude Modulation). Minimum Mean Square Error algorithm is one the most popular methods to detect MIMO but it requires matrix inversion, which can be computationally expensive for large MIMO systems. In the paper "*A Discrete First-Order Method for Large-Scale MIMO Detection with Provable Guarantees*", authors consider a simple and lowcomplexity discrete first-order method called the Generalized Power Method (GPM) for large-scale MIMO detection. This project focus on their approach (this is not their original code, you may find their source in ![https://huikang2019.github.io/code.html](https://huikang2019.github.io/code.html))

## Methodology

### Problem formulation
The goal is to detect transmitted symbols x in a MIMO system from the received signal: y = H @ x_true + v  # Received signal model (H: channel matrix, v: noise)
### Generalized Power Method (GPM) Algorithm
GPM is an iterative gradient-based detector that avoids matrix inversion (unlike MMSE). It works follow:
1. Gradient Calculation:
   ```python
   grad_F = 2 * H.conj().T @ (H @ x_prev - y)  # Gradient of the ML objective
   ```
2. Gradient Step:
   ```python
   z = x_prev - (alpha / m) * grad_F  # Step size scaled by alpha/m
   ```
3. Projection to QAM Constellation:
   ```python
   x_next = [project_to_constellation(z_i, constellation) for z_i in z]  # Nearest QAM point
   ```
4. Early Stopping:
   ```python
   if np.array_equal(x_next, x_prev): break  # Stop if no change
   ```
### QAM Constellation Handling

In MIMO detection, transmitted symbols are drawn from a finite set of complex values called a constellation, (e.g., 16-QAM). The detector must map noisy estimates back to valid constellation points.Firstly, generating the 16-QAM Constellation. Secondly, projecting noisy estimates to the constellation: fter gradient descent, the estimated symbols (z) may not align with valid QAM points. We need to calculate distances to every point in the constellation for each noisy estimate z (complex number). Then, return the constellation point with the smallest distance.

This is matter because in wireless communication the received signal y = Hx + v is corrupted by noise (v). After processing (e.g., MMSE or GPM), we get an estimate of the transmitted symbols, but these estimates are not perfect. And in order to ensure detected symbols are valid and minimizing Symbol Error Rate (SER), we need to constellation projection. And combination of gradient descent (for optimization) and QAM projection (for discrete constraints) makes GPM both theoretically sound and practical for real-world MIMO systems.

### Theoretical Guarantees of GPM for MIMO Detection (From the Paper)

The Generalized Power Method (GPM) is not just an empirical algorithm—it comes with strong theoretical guarantees that ensure its correctness and convergence under practical conditions. Here’s a detailed breakdown of the key theoretical insights from the paper:

1. Finite Convergence to the True Solution: traditional detectors (like MMSE) only minimize error statistically but cannot guarantee exact recovery. While GPM is guaranteed to recover the exact transmitted symbols x* in a finite number of steps. This is matter for high-reliability systems (e.g., 5G, IoT) where symbol errors are costly.
   
2. Condition 1: Small noise (Theorem 1)

   GPM converges if noise is small. The noise-corrupted gradient term must be smaller than half the minimum distance between QAM symbols. If noise is too large, it can "push" estimates into the wrong decision region.
   ```python
   term1 = np.abs((2 * alpha / m) * H.conj().T @ v)
    cond1 = np.max(term1) < (1 / c)  # c = 4/min_distance(QAM)
   ```
3. Condition 2: Well-Conditioned Channel (Theorem 2)

   The channel matrix H must be close to an identity transform (i.e., not too ill-conditioned). If H is poorly scaled (e.g., near-singular), gradient steps may diverge.
    ```python
    op_term = np.eye(n) - (2 * alpha / m) * H.conj().T @ H
    cond2 = np.linalg.norm(op_term, 2) < 0.25  # Operator norm check
    ```
## Installation
You may use tqdm package for solving optimization 
## Results 
You can test others cases or algorithms 

![](https://github.com/KingdomNguyen/image/blob/main/Screenshot%202025-07-28%20060741.jpg?raw=true)

## Reference
[1] H. Liu, M.-C. Yue, A. M.-C. So, and W.-K. Ma, “A discrete first-order
method for large-scale MIMO detection with provable guarantees,”
in Proc. IEEE Workshop Signal Process. Adv. Wireless Commun.
(SPAWC), Sapporo, Japan, Jul. 2017, pp. 669–673.

   

   
