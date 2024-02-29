from typing import List, TypeVar, Iterator
import random
from matplotlib import pyplot as plt

Vector = List[float]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]


def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


inputs = [(x, x * 20 + 5) for x in range(-50, 50)]


def _linear_gradient(x : float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta # наклон / угловой коэффициент kx+b (k)
    predicted = slope * x + intercept
    error = (predicted - y)
    #squared_error = error ** 2
    grad = [2 * error * x, 2 * error]
    return grad


# mini-batch gradient descent

learning_rate = 0.001

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

T = TypeVar('T')

def _minibatches(dataset: List[T], batch_size: int, shuffle: bool = True) -> Iterator[List[T]]:
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    if shuffle: # перетасовка
        random.shuffle(batch_starts)
    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

def _gradient_step(v : Vector, gradient : Vector, step_size : float) -> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

yarr = []

for epoch in range(1000):
    for batch in _minibatches(inputs, batch_size = 20):
        grad = vector_mean([_linear_gradient(x, y, theta) for x, y in batch])
        theta = _gradient_step(theta, grad, -learning_rate)
        yarr.append(theta[0])
    if epoch == 999:
        print(epoch, theta)

slope, intercept = theta

print("len yarr = ", len(yarr))

yarr1 = yarr[500:1000]
xarr = [x for x in range(500, 1000)]

assert len(xarr) == len(yarr1)
plt.title("Func y = 20x + 5")
plt.xlabel("iterate")
plt.ylabel("slope")
plt.scatter(xarr, yarr1, color = 'pink')
plt.show()
