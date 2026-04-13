I know how linear regression works but now we going to look at it, with the neural networks eye (:

this is what a linear model looks like
$$
\hat y = w_{1}x_{1} + ... + w_{d}x_{d} + b
$$
Now we put all the features and weight into a vector where $x \in \mathbb{R}^d$ and  $w \in \mathbb{R}^d$, we now express our model using a dot product:
$$
\hat{y} = w^{\top}x + b
$$
notice we took a a transpose of w, because the inner dimensions need to be the same for matrix multiplication

## [[linear regression]]
We need to define a measure of fitness in order for us to even think about fitting our model to data. you know what the loss function does.
There are different kinds of functions we could use to measure but for now we'll focus on the **[[Mean Squared Error (MSE)]]** loss function
$$
J_{(w, b)}^{(i)} = \frac{1}{2}(\hat{y}^{i} - y^{i})^2
$$
![[error_sketch .excalidraw]]

for our function to best fit the data we need to minimize the sum or the [[Mean Squared Error (MSE)]], this equation is called the **Loss function**
$$
L(w, b) = \frac{1}{n}\sum_{i = 1}^n J_{(w, b)}^{(i)} = \frac{1}{n} \sum_{i -1}^{n} \frac{1}{2} ((w^{\top}x^{i} + b) - y^{i})^{2}
$$
When we are training a a regression model we want to find parameters $(w^{*}, b^{*})$ that minimize the total loss across all the training examples:
$$
w^*, b^* = \underset{w,b}{argmin} \ L(w, b)
$$
## [[linear regression]]
I thought i knew this but i actually don't at all. this changed me a bit, because I've always wondered can't we just one shot finding the minimum because we already  know that the loss landscape has one global minimum. At the time i was studying for calculus test and i thought off level curves could help do this (:

The model we all know for linear regression can be written in another way:
$$
\hat{y} = w^\top x + b
\qquad\qquad
\hat{y} = w^\top x
$$
$$
\hat{y} = w_{1}x_{1} + b
\qquad \qquad
\hat{y} = w_{1}x_{1} + w_{2}x_{2} + w_{3}\cdot{1}
$$
The bias $b$ didn't disappear, we absorbed it into the weight vector. Notice that $b$ is just a constant, and $w3\cdot1$ is also just a constant. They play the same role. By appending a 1 to every input vector and treating the bias as another weight, we collapse the problem from finding two separate things ($w$ and $b$) into finding a single weight vector. This simplifies the math when we go to minimize the loss function directly.

So we now minimize our loss function with our now simpler equation.
OK now lasts do the math of how to actually derive the normal equation from minimizing the loss function

We want to minimize the sum of squared errors between predictions $\mathbf{Xw}$ and true targets $\mathbf{y}$:

$$ L(\mathbf{w}) = |\mathbf{Xw} - \mathbf{y}|^2 $$

> **Note:** We use $|\cdot|^2$ instead of MSE $= \frac{1}{n}|\cdot|^2$ because the $\frac{1}{n}$ is a constant that doesn't change where the minimum is. It cancels out when we set the derivative to zero.
 
The squared norm of any vector $\mathbf{v}$ is $\mathbf{v}^\top \mathbf{v}$. Here $\mathbf{v} = \mathbf{Xw} - \mathbf{y}$:

$$ L(\mathbf{w}) = (\mathbf{Xw} - \mathbf{y})^\top (\mathbf{Xw} - \mathbf{y}) $$

Expand like $(a - b)(a - b) = aa - ab - ba + bb$:

$$ = (\mathbf{Xw})^\top(\mathbf{Xw}) - (\mathbf{Xw})^\top \mathbf{y} - \mathbf{y}^\top(\mathbf{Xw}) + \mathbf{y}^\top \mathbf{y} $$

Substituting into the four terms:

$$ = \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w} - \mathbf{w}^\top \mathbf{X}^\top \mathbf{y} - \mathbf{y}^\top \mathbf{X} \mathbf{w} + \mathbf{y}^\top \mathbf{y} $$

Both $\mathbf{w}^\top \mathbf{X}^\top \mathbf{y}$ and $\mathbf{y}^\top \mathbf{X}\mathbf{w}$ are scalars (1×1 numbers). A scalar equals its own transpose:

$$ (\mathbf{w}^\top \mathbf{X}^\top \mathbf{y})^\top = \mathbf{y}^\top (\mathbf{X}^\top)^\top (\mathbf{w}^\top)^\top = \mathbf{y}^\top \mathbf{X} \mathbf{w} $$

Since they are the same number, we combine them:

$$ L(\mathbf{w}) = \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w} - 2\mathbf{w}^\top \mathbf{X}^\top \mathbf{y} + \mathbf{y}^\top \mathbf{y} $$


We now differentiate the loss function with respect to $\mathbf{w}$.

$$ \nabla_{\mathbf{w}} L = 2\mathbf{X}^\top \mathbf{X} \mathbf{w} - 2\mathbf{X}^\top \mathbf{y} + \mathbf{0} $$
Equate derivative to zero so to find value $\mathbf{w}$ that brings our cost function to a minimum
$$ 2\mathbf{X}^\top \mathbf{X} \mathbf{w} - 2\mathbf{X}^\top \mathbf{y} = \mathbf{0} $$


$$ \mathbf{X}^\top \mathbf{X} \mathbf{w} - \mathbf{X}^\top \mathbf{y} = \mathbf{0} $$


$$ \mathbf{X}^\top \mathbf{X} \mathbf{w} = \mathbf{X}^\top \mathbf{y} $$

This is the [[linear regression]].


$$ (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{X} \mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} $$

The left side simplifies: $(\mathbf{X}^\top \mathbf{X})^{-1}(\mathbf{X}^\top \mathbf{X}) = \mathbf{I}$ and $\mathbf{I}\mathbf{w} = \mathbf{w}$:

$$ \boxed{\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}} $$

This is the **closed-form (analytical) solution**. One formula, no iteration. Plug in your data matrix $\mathbf{X}$ and target vector $\mathbf{y}$, do the matrix math, and out comes the optimal weight vector.