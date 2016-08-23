#! /usr/bin/env python
""" Run one learning sequence, for a specific set of parameters. """
import time
import copy
import os
import sys

import numpy as np
import cPickle as pickle

import simulation
from basic_defs import *

##
## need a class to track progress
##

class LogProgress(object):

  """ A callable that stores progress information in a log file. """

  def __init__(self, logname, simulator, display=True, display_step=10):
    self.logname = logname
    self.f = None
    self.t0 = None
    self.display = display
    self.display_step = display_step
    self.simulator = simulator

  def __call__(self, i, n):
    t = time.time()
    if self.t0 is None:
      self.t0 = t

    t_diff = t - self.t0
    last_average_error = (self.simulator._current_res[-1]['average_error'] if 
      len(self.simulator._current_res) > 0 else np.inf)
    text = ('step: {} of {} ({}%); time elapsed: {:.1f}s ; error: {:.1f}'.
      format(i, n,  int(min(round(i*100.0/n),100)), t_diff, last_average_error))
    if self.f is None:
      if self.logname is not None:
        self.f = open(self.logname, 'w')

    if self.f is not None:
      self.f.write(text + '\n')
      self.f.flush()
    if self.display and (i%self.display_step == 0 or i == n):
      print(text)

def logProgressMaker(logname, **kwargs):
  def logProgressMakerWithLogname(simulator):
    return LogProgress(logname, simulator, **kwargs)

  return logProgressMakerWithLogname

def display_help():
  print("")
  print("Run the spiking simulation once and save the results.\n")
  print("Options:")
  print("  --defaults=<filename>    (mandatory)")
  print("      Python pickle file containing the default options to be used")
  print("      in the simulation, including target, tmax, and dt.")
  print("  --out=<filename>         (mandatory)")
  print("      File name where the script will pickle the results from the")
  print("      simulation. This will include a copy of the arguments used")
  print("      for the run.")
  print("  --nreps=<number>         (mandatory)")
  print("      Number of learning cycles to perform.")
  print("  --log=<filename>")
  print("      Name of log file. If not provided, there will be no log.")
  print("\n")
  print("All other command line arguments must also have the form")
  print("  --<key>=<value>")
  print("These will be directly passed to the SpikingLearningSimulation")
  print("constructor. Numeric values will automatically be transformed to")
  print("int (if possible) or float, and 'True' and 'False' will be changed")
  print("to their bool equivalents.\n")

if __name__ == '__main__':
  sim_args = {}
  for arg in sys.argv[1:]:
    if not arg.startswith('--'):
      print("Invalid command line option: " + arg + ".")
      display_help()
      sys.exit(1)

    # get rid of the double dash
    opt = arg[2:]

    # all options must have the form <name>=<value>
    key, sep, value = opt.partition('=')
    if len(sep) == 0:
      print("Invalid command line option: " + arg + ".")
      display_help()
      sys.exit(1)

    # get rid of extra spaces
    key = key.strip()
    value = value.strip()
    
    # get rid of surrounding quotes, if any
    value = value.strip('"\'')

    # convert to tuples/lists if necessary
    if ((value.startswith('(') and value.endswith(')')) or
        (value.startswith('[') and value.endswith(']'))):
      value = [float(_) for _ in value.strip('[()]').split(',')]
    else:
      # convert to bool or number, if possible
      if value == 'True':
        value = True
      elif value == 'False':
        value = False
      else:
        try:
          value = int(value)
        except ValueError:
          try:
            value = float(value)
          except ValueError:
            pass

    sim_args[key] = value

  # handle some special options, then send everything else to the simulation
  log_name = sim_args.pop('log', None)
  display = sim_args.pop('display', False)

  try:
    out_name = sim_args.pop('out')
  except KeyError:
    print("\nNeed an output file name (--out option).")
    display_help()
    sys.exit(1)

  try:
    defaults_name = sim_args.pop('defaults')
  except KeyError:
    print("\nNeed a defaults file name (--defaults option).")
    display_help()
    sys.exit(1)

  try:
    n_reps = sim_args.pop('nreps')
  except KeyError:
    print("\nNeed the number of repetitions (--nreps option).")
    display_help()
    sys.exit(1)

  # now start with the defaults, and update them with the command line options
  with open(defaults_name, 'rb') as inp:
    actual_args = pickle.load(inp)

  actual_args.update(sim_args)

  # add a progress indicator
  actual_args['progress_indicator'] = logProgressMaker(log_name,
      display=display)

  # need a tracker
  def tracker_generator(simulator, i, n):
    res = {}
    if i % 10 == 0:
      res['student'] = simulation.EventMonitor(simulator.student)
      res['motor'] = simulation.StateMonitor(simulator.motor, 'out')

    return res

  def snapshot_generator(simulator, i, n):
    res = {}
    if i % 50 == 0:
      res['weights'] = copy.copy(simulator.conductor_synapses.W)

    return res

  actual_args['tracker_generator'] = tracker_generator
  actual_args['snapshot_generator'] = snapshot_generator

  # run the simulation
  sim = SpikingLearningSimulation(**actual_args)
  res = sim.run(n_reps)

  # and save the results
  with open(out_name, 'wb') as out:
    actual_args.pop('progress_indicator', None)
    actual_args.pop('tracker_generator', None)
    actual_args.pop('snapshot_generator', None)

    pickle.dump({'args': actual_args, 'res': res}, out, 2)
