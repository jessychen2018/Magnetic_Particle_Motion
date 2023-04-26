"""
particle steering by randomized magnetic field
"""
import openpyxl as op
from matplotlib import pyplot as plt
import numpy as np
import pdb


class particle_motion:

    def __init__(self, network, boundary, T=500, dt=0.01):
        self.dim = 3  # dimension
        self.T, self.dt, self.dx = T, dt, 10**(-6)
        self.bg, self.boundary = network, boundary  # background (vascular network)
        self.r = 5e-6  # particle radius [m]
        self.ita_w = 1e-3  # viscosity of water
        self.Bx, self.By, self.Bz = 0.1, 0.1, 0.1  # magnetic field dB/dx, dB/dy, dB/dz [T/m]
        self.B0 = 0.5  # background magnetic field [T]
        self.mu0 = 4 * np.pi * 1e-7  # vacuum magnetic permeability
        self.mu_rp = 4.1  # relative magnetic permeability of particle
        self.pos = np.empty((int(self.T / self.dt), self.dim), dtype=np.float32)  # particle position, non-dimensional
        # self.pos[0, :] = [254,188,49]
        temp = np.where(self.bg == 255)
        rd = np.random.randint(0, np.shape(temp)[1])
        self.pos[0, :] = [temp[0][rd], temp[1][rd], temp[2][rd]]

        self.pulse_time, self.magnitude, self.energy = [], [], []

    def run(self, strategy, para1, para2):
        t_rec = 0   # record t
        vel, tp = self.micro_properties(0, strategy, para1, para2)
        for id in range(1, int(self.T / self.dt)):
            self.pos[id, :] = vel/self.dx*self.dt + self.pos[id-1, :]
            vel = self.rectify_motion(id, vel)
            t_rec += 1
            if t_rec > tp:
                vel, tp = self.micro_properties(id, strategy, para1, para2)
                t_rec = 0

    def micro_properties(self, id, strategy, para1, para2):
        direct = np.random.uniform(-1, 1, 3)  # randomly select a moving direction
        direct = direct / (np.sqrt(np.sum(direct ** 2)))

        if strategy == 'prw':
            A, tp = np.random.exponential(para1)*direct, np.random.exponential(para2)
        elif strategy == 'constant':
            A, tp = np.random.exponential(para1)*direct, para2
        elif strategy == 'levy':
            if para2 < 1 or para2 > 3:
                raise ValueError('Generalized levy walk only allow for 1.0 < mu < 3.0')
            A = np.random.exponential(para1)*direct
            tp = self.levy_rand(para2)
        elif strategy == 'nonlinear cor':
            A = np.random.exponential(para1)
            tp = np.exp(para2 * A) - 1
            A = A * direct
        else:
            raise ValueError('Input strategy does no exist...')

        H = (self.pos[id, :].copy()*self.dx*np.array([self.Bx, self.By, self.Bz])*A + np.array([self.B0, self.B0, self.B0]))/self.mu0
        Hg = np.array([self.Bx, self.By, self.Bz]) * A / self.mu0
        F_m = self.mu0 * 4 * np.pi * self.r**3 * (self.mu_rp-1)/(4*self.mu_rp-1) * H * Hg
        vel = F_m / (6 * np.pi * self.ita_w * self.r)

        self.pulse_time.append(id)
        self.magnitude.append(np.sqrt(np.sum(F_m**2))*10**(12))  # pN
        self.energy.append(0.5*self.mu0*np.sum(H**2)*10**(-6))  # change to pJ/um^3

        return vel, tp/self.dt

    def rectify_motion(self, id, vel):
        a, b, c = (np.round(self.pos[id])).astype(np.int32)

        if a>=self.boundary[0] or b>=self.boundary[1] or c>=self.boundary[2] or min(a, b, c) < 0:
            vel, ii = 0, 1
            while a>=self.boundary[0] or b>=self.boundary[1] or c>=self.boundary[2] or min(a, b, c) < 0:
                self.pos[id, :] = self.pos[id-ii, :]
                a, b, c = (np.round(self.pos[id])).astype(np.int32)
                ii += 1

        elif self.bg[a, b, c] <= 100:  # encounter boundary
            cube = self.bg[a-1:a+2, b-1:b+2, c-1:c+2]
            p_in = np.array(np.where(cube == 255))

            ii = 1
            while len(p_in[0]) <= 5 and self.bg[a, b, c] <= 100:
                self.pos[id, :] = self.pos[id-ii, :]
                a, b, c = (np.round(self.pos[id])).astype(np.int32)
                cube = self.bg[a-1:a+2, b-1:b+2, c-1:c+2]
                p_in = np.array(np.where(cube == 255))
                ii += 1
            p_r = int(np.random.uniform(0, len(p_in[0])))
            d_n = p_in[:, p_r] - np.array([1, 1, 1]) + np.random.uniform(-0.5, 0.5, 3)
            d_n = d_n / np.sqrt(np.sum(d_n*d_n))
            vel = np.sqrt(np.sum(vel*vel))*d_n

        return vel

    def levy_rand(self, para2, rd_min=0, rd_max=500):
        X = np.random.uniform(-np.pi / 2, np.pi / 2)
        Y = -np.log(np.random.uniform(0, 1))
        rd = np.sin((para2 - 1) * X) / np.cos(X) ** (1.0 / (para2 - 1)) * (
                np.cos((2 - para2) * X) / Y) ** ((2 - para2) / (1 - para2))
        while rd < rd_min or rd > rd_max:
            X = np.random.uniform(-np.pi / 2, np.pi / 2)
            Y = -np.log(np.random.uniform(0, 1))
            rd = np.sin((para2 - 1) * X) / np.cos(X) ** (1.0 / (para2 - 1)) * (
                    np.cos((2 - para2) * X) / Y) ** (
                         (2 - para2) / (1 - para2))
        return rd

    def clear_hist(self):
        self.pos = np.zeros((int(self.T / self.dt), 3), float)
        temp = np.where(self.bg == 255)
        rd = np.random.randint(0, np.shape(temp)[1])
        self.pos[0, :] = [temp[0][rd], temp[1][rd], temp[2][rd]]  # non-dimensional
        # print(self.pos[0, :])
        # self.pos[0, :] = [144, 100, 102]
        self.pulse_time, self.magnitude, self.energy = [], [], []





