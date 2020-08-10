import os
import math
import json
from common.numpy_fast import clip
# from common.realtime import sec_since_boot
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.lane_planner import eval_poly
import time
sec_since_boot = time.time

FT_TO_M = 0.3048

# by Zorrobyte
# version 4
# modified by ShaneSmiskol to add speed and curve direction as learning factors
# version 5 due to json incompatibilities

GATHER_DATA = True
VERSION = 5

def find_distance(pt1, pt2):
  x1, x2 = pt1[0], pt2[0]
  y1, y2 = pt1[1], pt2[1]
  return math.hypot(x2 - x1, y2 - y1)


class CurvatureLearner:
  def __init__(self):
    self.curvature_file = '/data/curvature_offsets.json'
    rate = 1 / 20.  # pathplanner is 20 hz
    self.learning_rate = 3.45e-3 * rate  # equivalent to x/12000
    self.write_frequency = 5  # in seconds
    self.min_lr_prob = .75
    self.min_speed = 15 * CV.MPH_TO_MS

    self.directions = ['left', 'right']
    self.cluster_coords = {'CLUSTER_0': [9.25841481, 0.08629771], 'CLUSTER_1': [12.86597836, 0.56739591], 'CLUSTER_2': [12.89578272, 0.10601015], 'CLUSTER_3': [17.27660487, 0.36905564], 'CLUSTER_4': [17.37257309, 0.08137747], 'CLUSTER_5': [22.67790441, 1.1858886], 'CLUSTER_6': [22.75900523, 0.08544827], 'CLUSTER_7': [23.34395146, 0.2972829], 'CLUSTER_8': [23.42699742, 0.82911176], 'CLUSTER_9': [23.43154885, 0.5181539], 'CLUSTER_10': [26.73458578, 0.12489106], 'CLUSTER_11': [29.3149337, 0.34829016]}
    self.y_axis_factor = 17.41918337  # weight y/curvature as much as speed
    self.min_curvature = 0.050916
    self._load_curvature()

  def group_into_cluster(self, v_ego, d_poly):
    TR = 0.9
    dist = v_ego * TR
    # we want curvature of road from start of path not car, so subtract d_poly[3]
    lat_pos = eval_poly(d_poly, dist) - d_poly[3]  # lateral position in meters at TR seconds
    closest_cluster = None

    if abs(lat_pos) >= self.min_curvature:
      sample_coord = [v_ego, abs(lat_pos * self.y_axis_factor)]

      dists = {cluster: find_distance(sample_coord, coord) for cluster, coord in self.cluster_coords.items()}
      closest_cluster = min(dists, key=dists.__getitem__)
    return closest_cluster, lat_pos

  def update(self, v_ego, d_poly, lane_probs, angle_steers):
    self._gather_data(v_ego, d_poly, angle_steers)
    offset = 0
    if v_ego < self.min_speed or math.isnan(d_poly[0]) or len(d_poly) != 4:
      return offset

    cluster, lat_pos = self.group_into_cluster(v_ego, d_poly)
    if cluster is not None:  # don't learn/return an offset if below min curvature
      direction = 'left' if lat_pos > 0 else 'right'
      lr_prob = lane_probs[0] + lane_probs[1] - lane_probs[0] * lane_probs[1]
      if lr_prob >= self.min_lr_prob:  # only learn when lane lines are present; still use existing offset
        learning_sign = 1 if lat_pos >= 0 else -1
        self.learned_offsets[direction][cluster] -= d_poly[3] * self.learning_rate * learning_sign  # the learning
      offset = self.learned_offsets[direction][cluster]

    self._write_curvature()
    return clip(offset, -0.3, 0.3)

  def _gather_data(self, v_ego, d_poly, angle_steers):
    if GATHER_DATA:
      with open('/data/curv_learner_data', 'a') as f:
        f.write('{}\n'.format({'v_ego': v_ego, 'd_poly': list(d_poly), 'angle_steers': angle_steers}))

  def _load_curvature(self):
    self._last_write_time = 0
    try:
      with open(self.curvature_file, 'r') as f:
        self.learned_offsets = json.load(f)
      print('read file')
      if 'version' in self.learned_offsets and self.learned_offsets['version'] == VERSION:
        print('file up to date!')
        return
    except:
      pass
    print('file NOT up to date! resetting')
    # can't read file, doesn't exist, or old version
    self.learned_offsets = {d: {c: 0. for c in self.cluster_coords} for d in self.directions}
    self._write_curvature()  # rewrite/create new file

  def _write_curvature(self):
    if sec_since_boot() - self._last_write_time >= self.write_frequency:
      with open(self.curvature_file, 'w') as f:
        f.write(json.dumps(self.learned_offsets, indent=2))
      os.chmod(self.curvature_file, 0o777)
      self._last_write_time = sec_since_boot()

CurvatureLearner()