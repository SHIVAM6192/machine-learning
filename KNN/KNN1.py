import math

# training data(x,y)
# x = [height, weight]
# y = label
data = [
    ([170, 70], "Male"),
    ([160, 60], "Female"),
    ([180, 80], "Male"),
    ([155, 55], "Female")
]

# Euclidean distance function
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# KNN algorithm
def knn_predict(test_point, k=3):
    distances = []
    for features, label in data:
        dist = euclidean_distance(test_point, features)
        distances.append((dist, label))
        
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    
    # Count votes
    votes = {}
    for _, label in neighbors:
        votes[label] = votes.get(label, 0) + 1
        
    prediction = max(votes, key=votes.get)
    return prediction

# Test the KNN algorithm
test = [165, 65]
retsult = knn_predict(test, k=3)
print(f"The predicted label for the test point {test} is: {retsult}")    

# Output: The predicted label for the test point [165, 65] is: Female