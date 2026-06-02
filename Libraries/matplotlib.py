
#! for ploting graph
import matplotlib.pyplot as plt
plt.scatter(X, y, color='red')
plt.plot(X, y, color='red')
plt.figure(figsize=(w,h))

plt.title('title')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.show()