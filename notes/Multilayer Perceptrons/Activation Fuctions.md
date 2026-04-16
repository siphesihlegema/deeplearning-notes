Activation functions are the to introduce non-linearity into the model. they also help with learning process by deciding whether a neuron should be activated or not by calculating the weighted sum and further adding bias with it.

### ReLu Function
Rectified linear unit (ReLu) provides a very simple nonlinear transformation. Given an element $x$, the function is defined as the maximum of the element and 0:
$$
ReLu(x) = \max{(x, 0)}
$$
In simple terms the ReLu function retains only positive elements and discards all negative elements by setting the corresponding activation to 0.
![[ReLuDrawing.excalidraw]]

![[Pasted image 20260416164401.png]]

This is the derivative of the ReLu function.