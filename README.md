# Method
1. Find time optimal trajectory duration via direct method.
2. Solve energy optimal trajectory optimisation problem with similar duration via indirect method.
3. Iteratively solve among cost homotopy until time optimal.
4. Segment nominal trajectory into nodes
5. Parallelly solve time optimal trajectory optimisation problems among state homotopy through multiple random walks about nominal time optimal trajectory.
5. Group trajectories with same task, and transcribe states relatively.
6. Train a neural network on state-control pairs for each task.
7. Encapsulate neural networks for each task within a task arbitrating behaviour tree.
8. Simualte.
