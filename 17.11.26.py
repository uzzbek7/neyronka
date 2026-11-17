import math
# 1
def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Векторы должны иметь одинаковую размерность")

    return sum(x * y for x, y in zip(a, b))




print(dot_product([1, 2, 3], [4, 5, 6]))
print(dot_product([1, 0], [0, 1]))
print(dot_product([-1, 2], [3, 4]))

try:
    print(dot_product([1, 2], [1, 2, 3]))
except ValueError as error:
    print(error)





# 2
def vector_norm(vector):
    return math.sqrt(sum(x ** 2 for x in vector))


def scale_score(score, d_k):
    if d_k <= 0:
        raise ValueError("Размерность d_k должна быть больше нуля")

    return score / math.sqrt(d_k)


vector = [3, 4]
print(vector_norm(vector))                # 5.0

score = 32
print(scale_score(score, 1))              # 32.0
print(scale_score(score, 4))              # 16.0
print(scale_score(score, 16))             # 8.0

try:
    print(scale_score(10, 0))
except ValueError as error:
    print(error)



# 3
def softmax(scores):
    if len(scores) == 0:
        raise ValueError("Список scores не должен быть пустым")

    max_score = max(scores)
    exponentials = [
        math.exp(score - max_score)
        for score in scores
    ]

    total = sum(exponentials)

    return [
        value / total
        for value in exponentials
    ]


scores = [1.0, 2.0, 3.0]
weights = softmax(scores)

print(weights)
print(sum(weights))
print(sum(weights) == 1.0 or math.isclose(sum(weights), 1.0))


print(weights[0] < weights[1] < weights[2])


# 3
def attention_scores(query, keys):
    if len(keys) == 0:
        raise ValueError("Список keys не должен быть пустым")
    query_dimension = len(query)
    for key in keys:
        if len(key) != query_dimension:
            raise ValueError("Размерность каждого ключа должна совпадать раазмерностью query")
    d_k = len(query)

    if d_k == 0:
        raise ValueError("Векторы не должны быть пустыми")
    scores = []
    for key in keys:
        score = dot_product(query, key)
        scaled_score = scale_score(score, d_k)
        scores.append(scaled_score)

    return scores

query = [1.0, 0.0]

keys = [
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5]
]

scores = attention_scores(query, keys)
print(scores)

 # 5
def attention_weights(query, keys):
    scores = attention_scores(query, keys)
    return softmax(scores)


weights = attention_weights(query, keys)

print(weights)
print(sum(weights))
print(math.isclose(sum(weights), 1.0))

print(weights[0] > weights[1])
print(weights[0] > weights[2])
