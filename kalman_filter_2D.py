import numpy as np


class KalmanFilter:

    def __init__(self, dt, x0, y0, vx0, vy0, a_psd, pos_unc, vel_unc, meas_noise_x, meas_noise_y, meas_cov):
        self.dt = dt
        self.x0 = x0
        self.y0 = y0
        self.vx0 = vx0
        self.vy0 = vy0
        self.a_psd = a_psd
        self.pos_unc = pos_unc
        self.vel_unc = vel_unc
        self.meas_noise_x = meas_noise_x
        self.meas_nouse_y = meas_noise_y
        self.meas_cov = meas_cov

        # create initial state estimate
        self.x = np.array([[self.x0], [self.y0], [self.vx0], [self.vy0]], dtype=float)
        # create transition matrix
        self.A = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        # since no acceleration values are provided assume that control matrix is null
        # create system noise covariance matrix
        self.Q = np.array([[(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0],
                           [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2],
                           [(self.dt ** 3) / 2, 0, self.dt ** 2, 0],
                           [0, (self.dt ** 3) / 2, 0, self.dt ** 2]]) * self.a_psd
        # create error covariance matrix
        self.P = np.array([[self.pos_unc ** 2, 0, 0, 0],
                           [0, self.pos_unc ** 2, 0, 0],
                           [0, 0, self.vel_unc ** 2, 0],
                           [0, 0, 0, self.vel_unc ** 2]])
        # create measurement noise covariance matrix
        self.R = np.array([[0.25, 0.1], [0.1, 0.25]])
        # create measurement matrix
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    def state_vector_estimate(self):
        # predict state time vector
        self.x = np.dot(self.A, self.x)
        # predict error covariance time propogation
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x

    def kalman_gain_update(self, x1, y1):
        # calculate Kalman Gain
        s = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        kg = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(s))
        # measurement
        z = np.array([[x1], [y1]])
        # update state vector
        self.x = np.round(self.x + np.dot(kg, (z - np.dot(self.H, self.x))), 2)
        # update error covariance matrix
        I = np.eye(self.H.shape[1])
        self.P = np.dot(I - (np.dot(kg, self.H)), self.P)
        return self.x, self.P
