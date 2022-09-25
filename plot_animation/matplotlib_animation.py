from matplotlib import pyplot as plt
from matplotlib import animation, rc
import numpy as np

# Static 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
xline = [*range(0, 6)]
yline = [*range(0, 6)]
zline = [*range(0, 6)]
ax.plot3D(xline, yline, zline)
plt.show()
plt.clf()

# Animation, run on Ipython
# Reference : http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-as-interactive-javascript-widgets/

from IPython.display import HTML

fig, ax = plt.subplots()
ax.set_xlim((0, 2))
ax.set_ylim((-2, 2))

# create animate elements
line, = ax.plot([], [],lw=2)
# Background plot each frame
def init():
    line.set_data([], [])
    return (line,)
# Animate function to be called each new frame 
def animate(i):
    x = np.linspace(0, 2, 10000)
    y = np.sin(2*np.pi*(x - 0.01*i))
    line.set_data(x, y)
    return (line,)
# Compile animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=20, 
                               blit=True)
HTML(anim.to_jshtml())

# Animation plot 3D, run from Ipython
def gen_rand_line(length, dims=2):
    """
    Create a line using a random walk algorithm.

    Parameters
    ----------
    length : int
        The number of points of the line.
    dims : int
        The number of dimensions of the line.
    """
    line_data = np.empty((dims, length))
    line_data[:, 0] = np.random.rand(dims)
    for index in range(1, length):
        # scaling the random numbers by 0.1 so
        # movement is small compared to position.
        # subtraction by 0.5 is to change the range to [-0.5, 0.5]
        # to allow a line to move backwards.
        step = (np.random.rand(dims) - 0.5) * 0.1
        line_data[:, index] = line_data[:, index - 1] + step
    return line_data


def update_lines(num, data_lines, lines):
    for line, data in zip(lines, data_lines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Fifty lines of random 3-D lines
data = [gen_rand_line(25, 3) for index in range(50)]

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d([0.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(
    fig, update_lines, 50, fargs=(data, lines), interval=50,blit=True)

HTML(line_ani.to_jshtml())