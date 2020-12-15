import numpy as np
from kalman_filter_2D import KalmanFilter


def main():
    # 2D position 4 state Kalman filter, assuming errors are the same along each axis and acceleration values are not
    # provided, pos_unc and vel_unc are sd and a_psd is var
    kevin = KalmanFilter(dt=1, x0=0, y0=0, vx0=0, vy0=0, a_psd=5, pos_unc=100, vel_unc=10, meas_noise_x=0.5, meas_noise_y=0.5, meas_cov=0.1)
    # t = 1
    x_predict = kevin.state_vector_estimate()
    x_estimate, updated_p = kevin.kalman_gain_update(-89.9, 210.2)
    print('@t=1')
    print(x_predict)
    print(x_estimate)
    print(updated_p)
    # t = 2
    x_predict = kevin.state_vector_estimate()
    x_estimate, updated_p = kevin.kalman_gain_update(-96.1, 204.9)
    print('@t=2')
    print(x_predict)
    print(x_estimate)
    print(updated_p)
    # t = 3
    x_predict = kevin.state_vector_estimate()
    x_estimate, updated_p = kevin.kalman_gain_update(-103.2, 201.3)
    print('@t=3')
    print(x_predict)
    print(x_estimate)
    print(updated_p)
    # t = 4
    x_predict = kevin.state_vector_estimate()
    x_estimate, updated_p = kevin.kalman_gain_update(-111.0, 197.9)
    print('@t=4')
    print(x_predict)
    print(x_estimate)
    print(updated_p)




if __name__ == "__main__":
    main()
