class LatControlPF:
  def __init__(self):
    # self.k_f = 0.00006908923778520113
    self.k_f = 0.00005
    self.speed = 0

  @property
  def k_p(self):
    return interp(self.speed, [20 * CV.MPH_TO_MS, 75 * CV.MPH_TO_MS], [.05, .15])

  def update(self, setpoint, measurement, speed):
    self.speed = speed
    # f = feedforward(setpoint, speed)
    f = standard_feedforward(setpoint, speed)
    # f = model_feedforward(setpoint, speed)

    error = setpoint - measurement

    p = error * self.k_p
    f = f * self.k_f

    left_x = np.array([-40, -10, 0])
    # left_x = np.array([-48, -12, -6, 0]) * np.interp(speed, [0, 80 * CV.MPH_TO_MS], [1.5, 1])
    left_y = np.array([0, 1, 1]) * np.interp(speed, [0, 80 * CV.MPH_TO_MS], [1, 0.5])
    # left_y = np.array([1, .9, .5, .25]) / 2

    # right_x = np.array([0, 6, 12, 48]) * np.interp(speed, [0, 80 * CV.MPH_TO_MS], [1.5, 1])
    right_x = np.array([0, 10, 20])
    # right_y = np.array([.25, .5, .9, 1])
    right_y = np.array([0, .25, 0]) * np.interp(abs(setpoint), [45, 90], [0, 1])
    # assert left_x[-1] == right_x[0]
    # assert left_y[-1] == right_y[0]

    right_mod = 1  # np.interp(abs(setpoint), [0, 90], [1, 0])
    left_mod = 1  # np.interp(abs(setpoint), [0, 90], [1, 2])

    if error < 0:
      p = p - (p * np.interp(error, left_x, left_y) * left_mod)
    elif error > 0:
      p = p - (p * np.interp(error, right_x, right_y) * right_mod)

    # y[0] = y[0] * np.interp(speed, [0, 80 * CV.MPH_TO_MS], [2.5, 1])
    # y[-1] = y[-1] * np.interp(speed, [0, 80 * CV.MPH_TO_MS], [2.5, 1])
    # p *= np.interp(abs(setpoint), [0, 90], [1, .5])
    ret = f + p

    return ret  # multiply by 1500 to get torque units
    # return np.clip(p + steer_feedforward, -1, 1)  # multiply by 1500 to get torque units


_pid = LatControlPF()
plot_response(angle=45, around=45, speed=45, _pid=_pid)
