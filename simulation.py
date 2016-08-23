""" Define a simple framework for time-evolving a set of arbitrary agents and
monitoring their evolution.
"""

import numpy as np

def int_r(f):
  """ Convert to nearest integer. """
  return int(np.round(f))


class Simulation(object):

  """ A class that manages the evolution of a set of agents.

  This is a simple objects that essentially just keeps track of simulation time
  and calls the `evolve(self, t, dt)` method on a set of agents, allowing them
  to update their states.

  There are a few bells and whistles. First of all, each agent can also have a
  method `prepare(self, tmax, dt)`. If this method exists, it is called before
  every `run` and can be used to prepare the agent for the simulation.

  Typically the agents are run in the order in which they are given as arguments
  to `__init__`. If, however, agents have a field called `order`, this is used
  to identify their position in the running hierarchy. Agents that don't have
  this field are assumed to have an order of 0.

  Attributes
  ----------
    agents: sequence
      The sequence of agents in the simulation. This is ordered according to the
      agents' `order` field (if it exists).
    dt: float
      Simulation time step.
  """

  def __init__(self, *agents, **kwargs):
    """ Initialize with a set of agents.

    Arguments
    ---------
      A1, A2, ...: agents
        These are the agents to be used in the simulation. Each agent should
        have a method `evolve(self, t, dt)` that is called for each time step.
        If the agent further has a method `prepare(self, tmax, dt)`, this is
        called before the simulation.
      dt: float (default: 0.1)
        Set the time step.
    """
    order = [getattr(agent, 'order', 0) for agent in agents]
    self.agents = [_[0] for _ in sorted(zip(agents, order), key=lambda x: x[1])]
    self.dt = kwargs.pop('dt', 0.1)

    if len(kwargs) > 0:
      raise TypeError("Unexpected keyword argument '" + str(kwargs.keys()[0]) +
        "'.")
    
  def run(self, t):
    """ Run the simulation for a time `t`. """
    # cache some values, for speed
    agents = self.agents
    dt = self.dt

    # prepare the agents that support it
    for agent in self.agents:
      if hasattr(agent, 'prepare'):
        agent.prepare(t, dt)

    # run the simulation
    crt_t = 0.0
    for i in xrange(int_r(t/dt)):
      for agent in agents:
        agent.evolve(crt_t, dt)

      crt_t += dt

class EventMonitor(object):

  """ A class that can be used to track agent 'events' -- effectively tracking a
  boolean vector from the target object.

  The `order` attribute for this class is set to 1 by default, so that it gets
  executed after all the usual agents are executed (so that events can be
  detected for the time step that just ended).

  Attributes
  ----------
    t: list
      Times at which events were registered.
    i: list
      Indices of units that triggered the events. This is matched with `t`.
    N: int
      Number of units in agent that is being tracked.
    agent:
      Agent that is being tracked.
    event: string
      The agent attribute that is being monitored.
  """

  def __init__(self, agent, event='spike'):
    """ Create a monitor.

    Arguments
    ---------
      agent:
        The agent whose events should be tracked.
      event: string
        Name of event to track. The agent should have an attribute with the name
        given by `event`, and this should be a sequence with a consistent size
        throughout the simulation.
    """
    self.event = event
    self.agent = agent

    self.t = []
    self.i = []

    self.order = 10

  def prepare(self, tmax, dt):
    self.t = []
    self.i = []
    self.N = None

  def evolve(self, t, dt):
    events = getattr(self.agent, self.event)
    if self.N is None:
      self.N = len(events)

    idxs = np.asarray(events).nonzero()[0]
    n = len(idxs)
    if n > 0:
      self.t.extend([t]*n)
      self.i.extend(idxs)

class StateMonitor(object):

  """ A class that can be used to monitor the time evolution of an attribute of
  an agent.

  The `order` attribute for this class is set to 1 by default, so that it gets
  executed after all the usual agents are executed. This means that it stores
  the values of the state variables at the end of each time step.

  Attributes
  ----------
    t: array
      Array of times where state has been monitored.
    <var1>:
    <var2>:
    ...
    <varK>: array, size (N, n)
      Values of monitored quantities. `N` is the number of units that are
      targeted, and `n` is the number of time steps.
    _agent:
      Agent that is being targeted.
    _interval: float
      Time interval used for recording.
    _targets: sequence of string
      Quantities to be recorded.
  """
  
  def __init__(self, agent, targets, interval=None):
    """ Create a state monitor.

    Arguments
    ---------
      agent:
        The agent whose attributes we're tracking.
      targets: string or iterable of strings.
        The names of the agent attribute(s)  that should be tracked.
      interval: float
        If provided, the interval of time at which to record. This should be an
        integer multiple of the simulation time step. If not provided, recording
        is done at every time step.
    """
    self._agent = agent
    self._interval = interval
    self._targets = [targets] if isinstance(targets, (str,unicode)) else targets
    self.order = 10

  def prepare(self, tmax, dt):
    if self._interval is None:
      self._interval = dt

    self._step = int_r(self._interval/dt)
    self.t = np.arange(0.0, tmax, self._step*dt)
    self._n = 0
    self._i = 0

    self._first_record = True

  def _prepare_buffers(self):
    """ Create recording buffers. """
    tgt_ptrs = []
    for tname in self._targets:
      target = getattr(self._agent, tname)
      dtype = getattr(target, 'dtype', type(target))
      # using Fortran ordering can make a huge difference in speed of monitoring
      # (factor of 2 or 3)!
      setattr(self, tname, np.zeros((np.size(target), len(self.t)), dtype=dtype,
              order='F'))
      
      # cache references to the targets, for faster access
      tgt_ptrs.append(getattr(self, tname))

    self._first_record = False
    self._target_ptrs = tgt_ptrs

  def evolve(self, t, dt):
    if self._n % self._step == 0:
      # make sure all buffers are the right size
      if self._first_record:
        self._prepare_buffers()

      agent = self._agent
      i = self._i
      for tname, storage in zip(self._targets, self._target_ptrs):
        target = getattr(agent, tname)
        storage[:, i] = target

      self._i += 1

    self._n += 1
