import numpy as np


# In[29]:


num_particles = 3
num_features = 5
max_iter = 10
w = 0.7
c1 = 2
c2 = 2
r1 = 0.5
r2 = 0.3

positions = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0],
    [1, 1, 0, 1, 0]
], dtype=float)

velocities = np.array([
    [0.2, -0.1, 0.3, -0.2, 0.1],
    [-0.1, 0.2, 0.1, -0.3, -0.2],
    [0.1, 0.1, -0.2, 0.3, -0.1]
], dtype=float)


fitness_values = np.array([0.85, 0.80, 0.78])

pbest_positions = positions.copy()
pbest_scores = fitness_values.copy()

gbest_index = np.argmax(pbest_scores)
gbest_position = pbest_positions[gbest_index].copy()
gbest_score = pbest_scores[gbest_index]

print("=== INITIALIZATION ===")
print("Positions:\n", positions)
print("Velocities:\n", velocities)
print("PBests:\n", pbest_positions)
print("GBest:", gbest_position, gbest_score, "\n")


# In[32]:


num_particles = 3
num_features = 5
max_iter = 10
w = 0.7
c1 = 2
c2 = 2
r1 = 0.5
r2 = 0.3

positions = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0],
    [1, 1, 0, 1, 0]
], dtype=float)

velocities = np.array([
    [0.2, -0.1, 0.3, -0.2, 0.1],
    [-0.1, 0.2, 0.1, -0.3, -0.2],
    [0.1, 0.1, -0.2, 0.3, -0.1]
], dtype=float)

fitness_values = np.array([0.85, 0.80, 0.78])

pbest_positions = positions.copy()
pbest_scores = fitness_values.copy()

gbest_index = np.argmax(pbest_scores)
gbest_position = pbest_positions[gbest_index].copy()
gbest_score = pbest_scores[gbest_index]

print("=== INITIALIZATION ===")
print("Positions:\n", positions)
print("Velocities:\n", velocities)
print("PBests:\n", pbest_positions)
print("GBest:", gbest_position, gbest_score, "\n")


# In[33]:


for iter in range(1, max_iter+1):
    print(f"===== ITERATION {iter} =====")
    for i in range(num_particles):
        velocities[i] = (
            w * velocities[i] +
            c1 * r1 * (pbest_positions[i] - positions[i]) +
            c2 * r2 * (gbest_position - positions[i])
        )
        positions[i] = positions[i] + velocities[i]
        positions[i] = np.clip(np.round(positions[i]), 0, 1)
        positions[i] = np.where(np.abs(positions[i]) < 1e-10, 0, positions[i])  # إزالة -0

    if iter < 4:
        fitness_values = np.array([0.85, 0.80, 0.78])
    else:
        fitness_values = np.array([0.87, 0.80, 0.78])

    for i in range(num_particles):
        if fitness_values[i] > pbest_scores[i]:
            pbest_scores[i] = fitness_values[i]
            pbest_positions[i] = positions[i].copy()

    gbest_index = np.argmax(pbest_scores)
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_score = pbest_scores[gbest_index]

    print("Positions:\n", positions)
    print("Velocities:\n", velocities)
    print("PBests:\n", pbest_positions)
    print("GBest:", gbest_position, gbest_score, "\n")


# In[ ]:




