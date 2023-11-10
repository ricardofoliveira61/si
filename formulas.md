# Formulas 

## Dense Layer

### Forward Propagation

The value of each output neuron can be calculated as the following:

$$
y_{j}=b_{j}+\sum_{i}x_{i}w_{i j}
$$

With matrices, we can compute this formula for every output neuron using a dot product :

$$
X=[x_{1}\ \ \cdots\ \ x_{i}]  \quad \quad W=\begin{bmatrix}
w_{11} & \cdots & w_{1i} \\
\vdots & \ddots & \vdots \\
w_{j1} & \cdots & w_{ji}
\end{bmatrix} \quad \quad B=\left[b_{1}\ \ \cdots\ \ b_{j}\right]
$$

$$
Y = XW + B
$$

### Backward Propagation

#### ∂E/∂X

First, we need to compute the derivative of the error with respect to the input (∂E/∂X). This will be the ∂E/∂Y for the layer before that one.

$$
\frac{\partial E}{\partial X}=\left[\frac{\partial E}{\partial x_{1}}\quad\frac{\partial E}{\partial x_{2}}\quad\ ...\quad\frac{\partial E}{\partial x_{i}}\right]
$$

Using the chain rule:

$$
{\frac{\partial E}{\partial x_{i}}}={\frac{\partial E}{\partial y_{1}}}{\frac{\partial y_{1}}{\partial x_{i}}}+\dots+{\frac{\partial E}{\partial y_{j}}}{\frac{\partial y_{j}}{\partial x_{i}}} 
$$

$$
=\frac{\partial E}{\partial y_{1}}w_{i1}+\ldots+\frac{\partial E}{\partial y_{j}}w_{i j}
$$

We can then write the whole matrix:

$$
\frac{\partial E}{\partial X}=\left[(\frac{\partial E}{\partial y_{1}}w_{11}+\dots+\frac{\partial E}{\partial y_{j}}w_{1j})\right.\ \dots\ \ \left.(\frac{\partial E}{\partial y_{1}}w_{i1}+\dots+\frac{\partial E}{\partial y_{j}}w_{i j})\right] 
$$

$$
=\left[{\frac{\partial E}{\partial y_{1}}}\quad ... \quad{\frac{\partial E}{\partial y_{j}}}\right] =\left[\frac{\partial E}{\partial y_{1}} ... \frac{\partial E}{\partial y_{j}}\right]
\begin{bmatrix}
w_{11} & \cdots & w_{1i} \\
\vdots & \ddots & \vdots \\
w_{j1} & \cdots & w_{ji}
\end{bmatrix}
$$

$$
={\frac{\partial E}{\partial Y}}W^{t}
$$

#### ∂E/∂W

To update the network weights, we need the error derivative with respect to every weight:

$$
\frac{\partial E}{\partial W}=
\begin{bmatrix}
\frac{\partial E}{\partial w_{11}} & \cdots & \frac{\partial E}{\partial w_{1j}} \\
\vdots & \ddots & \vdots \\
\frac{\partial E}{\partial w_{i1}} & \cdots & \frac{\partial E}{\partial w_{ij}}
\end{bmatrix}
$$

Using the chain rule:

$$
{\frac{\partial E}{\partial w_{i j}}}={\frac{\partial E}{\partial y_{1}}}{\frac{\partial y_{1}}{\partial w_{i j}}}+\dots+{\frac{\partial E}{\partial y_{j}}}{\frac{\partial y_{j}}{\partial w_{i j}}}
$$

$$
{}={\frac{\partial E}{\partial y_{j}}}x_{i}
$$

Therefore,

$$
\frac{\partial E}{\partial W}=
\begin{bmatrix}
\{\frac{\partial E}{\partial y_{1}}}x_{1} & \cdots & {\frac{\partial E}{\partial y_{j}}}x_{1} \\
\vdots & \ddots & \vdots \\
{\frac{\partial E}{\partial y_{1}}}x_{i} & \cdots & {\frac{\partial E}{\partial y_{j}}}x_{i}
\end{bmatrix}
$$

$$
{}=
\begin{bmatrix}
x_{1} \\
\vdots \\
x_{i}
\end{bmatrix}
\left[{\frac{\partial E}{\partial y_{1}}}\quad\quad...\quad\quad{\frac{\partial E}{\partial y_{j}}}\right]
$$

$$
=X^t\frac{\partial E}{\partial Y}
$$

#### ∂E/∂B

Now for the biases (one gradient per bias):

$$
{\frac{\partial E}{\partial B}}=\left[{\frac{\partial E}{\partial b_{1}}}\quad{\frac{\partial E}{\partial b_{2}}}\quad...\quad{\frac{\partial E}{\partial b_{j}}}\right]
$$

Again, using the chain rule:

$$
{\frac{\partial E}{\partial b_{j}}}={\frac{\partial E}{\partial y_{1}}}{\frac{\partial y_{1}}{\partial b_{j}}}+\dots+{\frac{\partial E}{\partial y_{j}}}{\frac{\partial y_{j}}{\partial b_{j}}}
$$

$$
={\frac{\partial E}{\partial y_{j}}}
$$

Therefore:

$$
\frac{\partial E}{\partial B}=\left[\frac{\partial E}{\partial y_{1}}\quad\frac{\partial E}{\partial y_{2}}\quad...\quad\frac{\partial E}{\partial y_{j}}\right]
$$

$$
={\frac{\partial E}{\partial Y}}
$$

Finally, we have the three formulas that we need for the backward propagation:

$$
\frac{\partial E}{\partial X}=\frac{\partial E}{\partial Y}W^{t}
$$

$$
\frac{\partial E}{\partial W}=X^{t}\frac{\partial E}{\partial Y}
$$

$$
\frac{\partial E}{\partial B}=\frac{\partial E}{\partial Y}
$$

### Gradient Descent

To update the weights:

$$
w_{i}\leftarrow w_{i}-\alpha\frac{\partial E}{\partial w_{i}}
$$

To update the biases:

$$
b_{i}\leftarrow b_{i}-\alpha\frac{\partial E}{\partial b_{i}}
$$