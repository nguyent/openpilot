from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad
from tensorflow.keras.models import Sequential, load_model
from scipy.optimize import curve_fit
import numpy as np
import random
import os
import tensorflow as tf
# import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dropout

from common.numpy_fast import interp
from torque_model.helpers import LatControlPF, TORQUE_SCALE, random_chance, STATS_KEYS, REVERSED_STATS_KEYS, MODEL_INPUTS, normalize_sample, normalize_value, feedforward, standard_feedforward
from torque_model.load import load_data
from sklearn.model_selection import train_test_split
from selfdrive.config import Conversions as CV
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.chdir('C:/Git/openpilot-repos/op-smiskol-torque/torque_model')

# print(tf.config.optimizer.get_experimental_options())
# tf.config.optimizer.set_experimental_options({'constant_folding': True, 'pin_to_host_optimization': True, 'loop_optimization': True, 'scoped_allocator_optimization': True})
# print(tf.config.optimizer.get_experimental_options())

try:
  tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

to_normalize = False
data, data_sequences, data_stats, _ = load_data(to_normalize)
# del data_high_delay, data_sequences
print(f'Number of samples: {len(data)}')

x_train = []
for line in data:
  x_train.append([line[inp] for inp in MODEL_INPUTS])

y_train = []
for line in data:
  # the torque key is set by the data loader, it can come from torque_eps or torque_cmd depending on engaged status
  y_train.append(line['torque'])

print(f'Output (torque) min/max: {[min(y_train), max(y_train)]}')

# x_train = []  # only use synthetic samples
# y_train = []

# sns.distplot([abs(line['steering_angle']) for line in data], bins=200)
# plt.title('steering angle')
# plt.pause(0.01)
# input()
# plt.clf()
# sns.distplot([abs(line['steering_rate']) for line in data], bins=200)
# plt.title('steering rate')
# plt.pause(0.01)
# input()
# plt.clf()
# sns.distplot([line['v_ego'] for line in data], bins=200)
# plt.title('speed')
# plt.pause(0.01)
# input()
# plt.clf()
# sns.distplot([abs(line['torque']) for line in data], bins=200)
# plt.title('torque')
# plt.pause(0.01)
# input()


x_train = np.array(x_train)
y_train = np.array(y_train) / TORQUE_SCALE

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)
print('Training on {} samples and validating on {} samples'.format(len(x_train), len(x_test)))


# model = load_model('models/fifth_model.h5', custom_objects={'LeakyReLU': LeakyReLU})

model = Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Dense(8, activation=LeakyReLU()))
# model.add(Dropout(1/8))
model.add(Dense(16, activation=LeakyReLU()))
# model.add(Dropout(1/16))
# model.add(Dense(24, activation=LeakyReLU()))
model.add(Dense(1))

epochs = 150
starting_lr = .005
ending_lr = 0.001
decay = (starting_lr - ending_lr) / epochs

opt = Adam(learning_rate=starting_lr, amsgrad=True, decay=decay)
# opt = Adadelta(learning_rate=0.001)
# opt = Adagrad(learning_rate=0.2)
model.compile(opt, loss='mae', metrics='mse')
try:
  model.fit(x_train, y_train, batch_size=1024, epochs=100, validation_data=(x_test, y_test))
  model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=128, epochs=25, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=32, epochs=25, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))
except KeyboardInterrupt:
  pass

pid = LatControlPF()


def plot_response(angle=15, around=15, speed=37, _pid=pid):  # plots model output compared to pid on steady angle but changing desired angle
  # the two lines should ideally be pretty close
  plt.figure(2)
  plt.clf()
  desired = np.linspace(angle - around, angle + around, 100)
  error = np.array(desired) - angle
  rate = normalize_value(0, 'rate', data_stats, to_normalize)
  speed *= CV.MPH_TO_MS

  similar_data = [line for line in data if abs(line['steering_angle'] - angle) < 5 and abs(line['v_ego'] - speed) < 5]
  # for line in similar_data:
  #   line['error'] =
  data_error = [line['fut_steering_angle'] - line['steering_angle'] for line in similar_data]
  data_torque = [line['torque'] for line in similar_data]

  y_pid = []
  y_pid_new = []
  y_model = []
  for des in desired:
    pred = model.predict_on_batch(np.array([[normalize_value(des, "angle", data_stats, to_normalize), normalize_value(angle, "angle", data_stats, to_normalize), rate, rate, normalize_value(speed, "speed", data_stats, to_normalize)]]))[0][0]
    y_model.append(pred * 1500)
    y_pid.append(pid.update(des, angle, speed) * 1500)
    y_pid_new.append(_pid.update(des, angle, speed) * 1500)
  plt.plot(error, y_pid, label='standard pf controller')
  plt.plot(error, y_pid_new, label='new pf controller')
  plt.plot(error, y_model, label='model')
  plt.scatter(data_error, data_torque, label='data', s=1)
  plt.plot([0] * len(y_pid), np.linspace(max(y_model), min(y_model), len(y_pid)))
  plt.xlabel('angle error')
  plt.ylabel('torque')
  plt.legend()
  plt.show()
  return y_model, desired, angle, speed


def model_to_poly(x, _c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10):  # , _c10, _c11):  #, _c12):
  des_angle, angle, speed = x.copy()

  ret = []
  for setpoint, measurement, _speed in zip(des_angle, angle, speed):
    error = setpoint - measurement
    # if measurement * setpoint > 0 and (error * measurement < 0 or error * setpoint > 0):
    # if (measurement < 0 and (setpoint > 0 or setpoint > measurement) and error > 0) and (not (measurement < 0 and setpoint < 0 and error > 0) or setpoint > measurement):
    if abs(setpoint) < abs(measurement) or setpoint * measurement < 0:
      mod = _c1 * abs(error)
      mod += _c3 * abs(setpoint)
      mod += _c2
      # mod += _c9 * _speed ** 2 + _c11 * _speed
    else:
      mod = _c4 * abs(error)
      mod += _c6 * abs(setpoint)
      mod += _c5
      # mod += _c10 * _speed ** 2 + _c12 * _speed

    weight = np.interp(abs(error), [0, _c7 * abs(measurement) + _c8], [1, 0])
    # weight = np.interp(abs(error), [0, 30], [1, 0])

    p = error * interp(_speed, [20 * CV.MPH_TO_MS, 70 * CV.MPH_TO_MS], [.05, .15])
    f = feedforward(setpoint, _speed) * 3.768382789259873e-05
    # f = standard_feedforward(setpoint, _speed) * 0.00005
    # f = (_c9 * _speed ** 2 + _c10 * _speed + _c11) * setpoint

    new_p = p + p * mod
    p = (new_p * weight) + ((1 - weight) * p)

    p *= (_c9 * _speed + _c10)

    ret.append((p + f) * 1500)
  return ret


runs = []
# runs.append(plot_response(angle=0, around=15, speed=80))
# runs.append(plot_response(angle=2, around=35, speed=72))
# runs.append(plot_response(angle=7, around=20, speed=65))
runs.append(plot_response(angle=0, around=20, speed=72))
runs.append(plot_response(angle=35, around=25, speed=35))
runs.append(plot_response(angle=25, around=30, speed=40))
# runs.append(plot_response(angle=-35, around=25, speed=35))
# runs.append(plot_response(angle=90, around=45, speed=15))
runs.append(plot_response(angle=90, around=45, speed=35))
runs.append(plot_response(angle=180, around=20, speed=30))
# runs.append(plot_response(angle=25, around=15, speed=35))
# runs.append(plot_response(angle=5, around=10, speed=35))
# runs.append(plot_response(angle=10, around=15, speed=35))

x_train = []
y_train = []
for run in runs:
  y_model, desired_list, angle, speed = run
  for _des_angle, _y in zip(desired_list, y_model):
    x_train.append([_des_angle, angle, speed])
    y_train.append(_y)



params, _ = curve_fit(model_to_poly, np.array(x_train).T, y_train)
params = params.tolist()
print(params)


def plot_random_samples():
  idxs = np.random.choice(range(len(x_train)), 50)
  x_test = x_train[idxs]
  y_test = y_train[idxs].reshape(-1) * TORQUE_SCALE
  pred = model.predict(np.array([x_test])).reshape(-1) * TORQUE_SCALE

  plt.figure(0)
  plt.clf()
  plt.plot(y_test, label='ground truth')
  plt.plot(pred, label='prediction')
  plt.legend()
  plt.show()


# plot_random_samples()




def plot_static_error_response(error=5, angle_from=45, angle_to=40, speed=37):  # error will be added to des_angle
  # the two lines should ideally be pretty close
  plt.figure(3)
  plt.clf()
  angles = np.linspace(angle_from, angle_to, 200)
  rate = normalize_value(0, 'rate', data_stats, to_normalize)
  speed *= CV.MPH_TO_MS
  y_pid = []
  y_model = []
  for ang in angles:
    y_model.append(
      model.predict_on_batch(np.array([[normalize_value(ang + error, "angle", data_stats, to_normalize), normalize_value(ang, "angle", data_stats, to_normalize), rate, rate, normalize_value(speed, "speed", data_stats, to_normalize)]]))[0][0] * 1500)
    y_pid.append(pid.update(ang + error, ang, speed) * 1500)
  plt.plot(angles, y_pid, label='standard pf controller')
  plt.plot(angles, y_model, label='model')
  plt.plot([0] * len(y_pid), np.linspace(max(y_model), min(y_model), len(y_pid)))
  plt.xlabel('angle error')
  plt.ylabel('torque')
  plt.legend()
  plt.show()


def plot_moving_error_response(desired=0, around=15, speed=37):  # ignores feedforward component and tries to plot how the model responds to different errors (it's super smooth in ramping down)
  # the two lines should ideally be pretty close
  plt.figure(3)
  plt.clf()
  angle = np.linspace(desired - around, desired + around, 200)
  error = np.array(desired) - angle
  rate = normalize_value(0, 'rate', data_stats, to_normalize)
  speed *= CV.MPH_TO_MS
  y_pid = []
  y_model = []
  for ang in angle:
    y_model.append(
      model.predict_on_batch(np.array([[normalize_value(desired, "angle", data_stats, to_normalize), normalize_value(ang, "angle", data_stats, to_normalize), rate, rate, normalize_value(speed, "speed", data_stats, to_normalize)]]))[0][0] * 1500)
    y_pid.append(pid.update(desired, ang, speed) * 1500)
  plt.plot(error, y_pid, label='standard pf controller')
  plt.plot(error, y_model, label='model')
  plt.plot([0] * len(y_pid), np.linspace(max(y_model), min(y_model), len(y_pid)))
  plt.xlabel('angle error')
  plt.ylabel('torque')
  plt.legend()
  plt.show()


plot_idxs = {}
last_idx = 0
def plot_sequence(sequence_idx=3, show_controller=True, _pid=pid):  # plots what model would do in a sequence of data
  global last_idx
  sequence = data_sequences[sequence_idx]

  if sequence_idx not in plot_idxs:
    plot_idxs[sequence_idx] = last_idx
    last_idx += 1
  plt.figure(plot_idxs[sequence_idx])
  plt.clf()

  ground_truth = [line['torque'] for line in sequence]
  plt.plot(ground_truth, label='ground truth')

  USE_RATES = False

  _x = [normalize_sample(line, data_stats, to_normalize) for line in sequence]
  _x = [[(line[inp] if ('rate' in inp and USE_RATES or 'rate' not in inp) else 0) for inp in MODEL_INPUTS] for line in _x]
  pred = model.predict(np.array(_x)).reshape(-1) * TORQUE_SCALE
  plt.plot(pred, label='prediction')

  if show_controller:
    controller = [pid.update(line['fut_steering_angle'], line['steering_angle'], line['v_ego']) * TORQUE_SCALE for line in sequence]  # what a pf controller would output
    plt.plot(controller, linestyle=':', label='standard controller')

    # controller_v2 = [_pid.update(line['fut_steering_angle'], line['steering_angle'], line['v_ego']) * TORQUE_SCALE for line in sequence]  # what a pf controller would output
    # plt.plot(controller_v2, color='purple', label='new controller')

  plt.legend()
  plt.show()


def plot_seqs(_pid=pid):
  plot_sequence(-3, _pid=_pid)
  plot_sequence(6, _pid=_pid)
  plot_sequence(7, _pid=_pid)
  plot_sequence(59, _pid=_pid)
  # plot_sequence(46)
  # plot_sequence(15)

plot_seqs()
