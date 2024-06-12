import argparse
import numpy as np
import scipy
import scipy.fftpack
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import animation

# Defining the problem constants
xright = 2
yright = 2
M = 128
N = 128
x = np.linspace(0.5*xright/M, xright-0.5*xright/M, M)
y = np.linspace(0.5*yright/N, yright-0.5*yright/N, N)
h = x[1]-x[0]
X, Y = np.meshgrid(x, y, indexing='ij')
epsilon = 4*h/(2*np.sqrt(2)*np.arctanh(0.9))
xp = np.linspace(0, (M-1)/xright, M)
yq = np.linspace(0, (N-1)/yright, N)


def dct2(a: np.ndarray) -> np.ndarray:
    '''
    from https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
    '''
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a: np.ndarray) -> np.ndarray:
    '''
    from https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
    '''
    return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


class CahnHilliardSimulator:
    '''
    Simulates an example Cahn-Hilliard system. The implementation is based on

    Lee, D., Huh, J. Y., Jeong, D., Shin, J., Yun, A., & Kim, J. (2014). 
    Physical, mathematical, and numerical derivations of the Cahn–Hilliard equation. 
    Computational Materials Science, 81, 216-225.
    '''
    def __init__(self, u: np.ndarray, dt: float, iters: int):
        self.u = u
        self.t = 0.0
        self.dt = dt
        self.iters = iters
        self.f = lambda a: a**3 - 3*a
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.Leig = -(np.tile((xp**2), (N, 1)).T + np.tile(yq**2, (M, 1)))*(np.pi**2)
        self.CHeig = np.ones((M, N)) - 2*self.dt*self.Leig + self.dt*(epsilon**2)*(self.Leig**2)

    def step(self) -> np.ndarray:
        fu = self.f(self.u)
        hat_u = dct2(np.real(self.u))
        df = dct2(np.real(fu))
        hat_u = (hat_u + self.dt*self.Leig*df) / self.CHeig
        self.u = idct2(hat_u)
        self.t += self.dt
        return self.u


def main(args: argparse):
    np.random.seed(seed=2024)
    u0 = args.a + args.b * (np.random.rand(M, N)-0.5)
    simulator = CahnHilliardSimulator(u=u0, dt=args.dt, iters=args.iters)

    states = [u0]

    for _ in range(args.iters):
        states.append(simulator.step())

    if args.anim:
        cmap = mpl.colormaps.get_cmap('bwr')
        normalizer = Normalize(-1.2, 1.2)
        fig, ax = plt.subplots()
        
        def animate(j):
            ax.contourf(simulator.X, simulator.Y, states[j], cmap=cmap, norm=normalizer)
            ax.set_title(f't={simulator.dt*j:0.3f}')
            ax.set_xticks([])
            ax.set_yticks([])
        # Render video
        anim = animation.FuncAnimation(fig, animate, frames=range(args.iters), interval=20)
        # writervideo = animation.FFMpegWriter(fps=args.animfps)
        # anim.save(args.animname, writer=writervideo)
        anim.save(args.animname)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run example to illustrate the Laplace Proximal Point algorithm')
    parser.add_argument('--a', type=float, default=-0.45, help='initial state mean')
    parser.add_argument('--b', type=float, default=0.1, help='initial state standard deviation')
    parser.add_argument('--dt', type=float, default=0.01, help='time step size')
    parser.add_argument('--iters', type=int, default=500, help='maximum number of iterations')
    parser.add_argument('--anim', type=bool, default=True, help='animate results')
    parser.add_argument('--animfps', type=int, default=15, help='frames per second')
    parser.add_argument('--animname', type=str, default='result.gif', help='name of animation file')
    args = parser.parse_args()
    main(args)
