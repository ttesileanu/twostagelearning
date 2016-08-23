#! /usr/bin/env python

import simulation
import unittest

import numpy as np
import copy

from basic_defs import *

def int_r(f):
  """ Convert to nearest integer. """
  return int(np.round(f))

# a simple mock source layer
class SimpleNeurons(object):
  def __init__(self, N, out_step=1, out_fct=None):
    self.N = N
    self.out_step = out_step
    self.out_fct = out_fct

  def prepare(self, t_max, dt):
    if self.out_fct is None:
      if np.size(self.out_step) == 1:
        self.out = self.out_step*np.ones(self.N)
      else:
        self.out = np.copy(self.out_step)
    else:
      self.out = self.out_fct(0)
    self.i = 0
  
  def evolve(self, t, dt):
    if self.out_fct is None:
      self.out += self.out_step
    else:
      self.out = self.out_fct(self.i)
    self.i += 1

##################
# TestSimulation #
##################

class TestSimulation(unittest.TestCase):
  def test_creation(self):
    """ Create a simulation. """
    sim = simulation.Simulation(1, 2, [1, 2])
    self.assertSequenceEqual(sim.agents, [1, 2, [1, 2]])

  def test_number_of_steps(self):
    """ Test the number of steps executed upon run. """
    class Mock(object):
      def __init__(self):
        self.count = 0

      def evolve(self, t, dt):
        self.count += 1

    G = Mock()
    sim = simulation.Simulation(G, dt=0.1)

    sim.run(100.0)
    self.assertEqual(G.count, 1000)

    G = Mock()
    sim = simulation.Simulation(G, dt=0.2)
    sim.run(100.2)
    self.assertEqual(G.count, 501)

  def test_prepare_on_run(self):
    """ Ensure that the agents' `prepare` method is called on `run()`. """
    class Mock(object):
      def __init__(self):
        self.t_max = None
        self.dt = None

      def evolve(self, t, dt):
        pass

      def prepare(self, t_max, dt):
        self.t_max = t_max
        self.dt = dt

    t_max = 10.0
    dt = 0.2
      
    G = Mock()
    sim = simulation.Simulation(G, dt=dt)
    self.assertIsNone(G.t_max)
    self.assertIsNone(G.dt)

    sim.run(t_max)
    self.assertEqual(G.t_max, t_max)
    self.assertEqual(G.dt, dt)

  def test_evolution(self):
    """ Test time-evolution in a simulation. """
    class Mock(object):
      def __init__(self):
        self.t = 0.0

      def evolve(self1, t, dt):
        self.assertEqual(self1.t, t)
        self1.t += dt

    G1 = Mock()
    G2 = Mock()
    sim = simulation.Simulation(G1, G2)
    sim.run(100.0)

    self.assertAlmostEqual(G1.t, 100.0)

  def test_order(self):
    """ Test that the agents are evolved in the correct order. """
    n = 3
    was_called = n*[False]
    class Mock(object):
      def __init__(self, i):
        self.i = i

      def evolve(self1, t, dt):
        was_called[self1.i] = True
        self.assertTrue(all(was_called[:self1.i]))

    sim = simulation.Simulation(*[Mock(_) for _ in xrange(n)])
    sim.run(sim.dt)

  def test_custom_order_exec(self):
    """ Test that the execution order of the agents can be modified. """
    self.call_idx = 0
    class Mock(object):
      def __init__(self, order):
        self.order = order
        self.execd = -1

      def evolve(self1, t, dt):
        self1.execd = self.call_idx
        self.call_idx += 1

    G1 = Mock(3)
    G2 = Mock(-1)
    G3 = Mock(2)
    sim = simulation.Simulation(G1, G2, G3)
    sim.run(sim.dt)

    self.assertEqual(G1.execd, 2)
    self.assertEqual(G2.execd, 0)
    self.assertEqual(G3.execd, 1)

  def test_custom_order_prepare(self):
    """ Test that the preparation order of the agents can be modified. """
    self.call_idx = 0
    class Mock(object):
      def __init__(self, order):
        self.order = order
        self.execd = -1

      def evolve(self, t, dt):
        pass

      def prepare(self1, t_max, dt):
        self1.execd = self.call_idx
        self.call_idx += 1

    G1 = Mock(1)
    G2 = Mock(10)
    G3 = Mock(-5)
    sim = simulation.Simulation(G1, G2, G3)
    sim.run(sim.dt)

    self.assertEqual(G1.execd, 1)
    self.assertEqual(G2.execd, 2)
    self.assertEqual(G3.execd, 0)

  def test_timestep(self):
    """ Test changing the simulation time step. """
    class Mock(object):
      def __init__(self):
        self.t = 0.0
        self.dt = None

      def evolve(self1, t, dt):
        if self1.dt is not None:
          self.assertAlmostEqual(self1.dt, dt)
        else:
          self1.dt = dt

        self.assertAlmostEqual(self1.t, t)

        self1.t += self1.dt

    t_max = 10.0
    dt = 0.2

    G = Mock()
    simulation.Simulation(G, dt=dt).run(t_max)
    self.assertAlmostEqual(G.dt, dt)


####################
# TestEventMonitor #
####################

class TestEventMonitor(unittest.TestCase):
  def setUp(self):
    # generate pseudo-random test case
    np.random.seed(123456)
    class Spiker(object):
      def __init__(self, pattern):
        self.N = pattern.shape[0]
        self.pattern = pattern
        self.i = 0
        self.spike = np.zeros(self.N, dtype=bool)
        self.other = np.zeros(self.N, dtype=bool)

      def evolve(self, t, dt):
        self.spike = self.pattern[:, self.i]
        self.other = ~self.pattern[:, self.i]
        self.i += 1

    self.t_max = 16.0  # duration of simulation
    self.dt = 2.0     # time step
    self.N = 15       # number of units
    self.p = 0.2      # probability of spiking

    self.G = Spiker(np.random.rand(self.N, int_r(self.t_max/self.dt)) < self.p)

  def test_init(self):
    """ Test that monitor is properly initialized. """
    M = simulation.EventMonitor(self.G)
    self.assertTrue(hasattr(M, 't'))
    self.assertTrue(hasattr(M, 'i'))

    self.assertEqual(len(M.t), 0)
    self.assertEqual(len(M.i), 0)

  def test_shape(self):
    """ Test that time and index vectors have matching lengths. """
    M = simulation.EventMonitor(self.G)
    sim = simulation.Simulation(self.G, M, dt=self.dt)
    sim.run(self.t_max)

    self.assertEqual(len(M.t), len(M.i))

  def test_spike_order(self):
    """ Test that the spike times are ordered in ascending order. """
    M = simulation.EventMonitor(self.G)
    sim = simulation.Simulation(self.G, M, dt=self.dt)
    sim.run(self.t_max)

    self.assertNotEqual(len(M.t), 0)
    self.assertTrue(all(M.t[i] <= M.t[i+1] for i in xrange(len(M.t) - 1)))

  def test_order(self):
    """ Test that by default the monitor is called after its target. """
    class Mock(object):
      def __init__(self):
        self.spike = [False]

      def evolve(self, t, dt):
        self.spike = [True]

    G0 = Mock()
    M0 = simulation.EventMonitor(G0)
    sim = simulation.Simulation(G0, M0)
    sim.run(sim.dt)

    self.assertEqual(len(M0.t), 1)
    self.assertEqual(len(M0.i), 1)

  def test_accurate(self):
    """ Test that all the events are properly stored. """
    M = simulation.EventMonitor(self.G)
    sim = simulation.Simulation(self.G, M, dt=self.dt)
    sim.run(self.t_max)

    times = self.G.pattern.nonzero()[1]*self.dt
    self.assertTrue(np.allclose(sorted(times), M.t))
    for (i, t) in zip(M.i, M.t):
      self.assertTrue(self.G.pattern[i, int_r(t/self.dt)])

  def test_other_event(self):
    """ Test following a different event. """
    M = simulation.EventMonitor(self.G, event='other')
    sim = simulation.Simulation(self.G, M, dt=self.dt)
    sim.run(self.t_max)

    times = (~self.G.pattern).nonzero()[1]*self.dt
    self.assertTrue(np.allclose(sorted(times), M.t))
    for (i, t) in zip(M.i, M.t):
      self.assertFalse(self.G.pattern[i, int_r(t/self.dt)])


####################
# TestStateMonitor #
####################

class TestStateMonitor(unittest.TestCase):
  def setUp(self):
    # make something to track
    class Mock(object):
      def __init__(self, N):
        self.N = N
        self.v = np.zeros(N)
        self.a = np.zeros(2)
        self.b = np.zeros(N, dtype=bool)
        self.f = 0.0

      def evolve(self, t, dt):
        self.v += np.arange(self.N)*dt
        self.a += dt
        self.b = ((np.arange(self.N) + int_r(t/dt)) % 3 ==0)
        self.f = t

    self.N = 15
    self.t_max = 10.0
    self.dt = 0.1
    self.G = Mock(self.N)

  def test_time(self):
    """ Test that time is stored properly. """
    M = simulation.StateMonitor(self.G, 'v')
    sim = simulation.Simulation(self.G, M, dt=self.dt)
    sim.run(self.t_max)
    self.assertTrue(np.allclose(M.t, np.arange(0, self.t_max, self.dt)))

  def test_custom_time(self):
    """ Test that time is stored properly with custom interval. """
    interval = 0.5
    M = simulation.StateMonitor(self.G, 'v', interval=interval)
    sim = simulation.Simulation(self.G, M, dt=self.dt)
    sim.run(self.t_max)
    self.assertTrue(np.allclose(M.t, np.arange(0, self.t_max, interval)))

  def test_shape(self):
    """ Test that the shape of the data storage is correct. """
    M = simulation.StateMonitor(self.G, ['a', 'v'])
    sim = simulation.Simulation(self.G, M, dt=self.dt)
    sim.run(self.t_max)

    nsteps = int_r(self.t_max/self.dt)
    self.assertEqual(M.v.shape, (self.N, nsteps))
    self.assertEqual(M.a.shape, (2, nsteps))

  def test_shape_interval(self):
    """ Test correct shape with custom interval. """
    interval = 0.5
    M = simulation.StateMonitor(self.G, ['a', 'v', 'b'], interval=interval)
    sim = simulation.Simulation(self.G, M, dt=self.dt)
    sim.run(self.t_max)

    nsteps = int_r(self.t_max/interval)
    self.assertEqual(M.v.shape, (self.N, nsteps))
    self.assertEqual(M.a.shape, (2, nsteps))
    self.assertEqual(M.b.shape, (self.N, nsteps))

  def test_1d_function(self):
    """ Test storage for 1d function. """
    M = simulation.StateMonitor(self.G, 'f')
    sim = simulation.Simulation(self.G, M, dt=self.dt)
    sim.run(self.t_max)
    self.assertTrue(np.allclose(M.f, M.t))

  def test_accurate(self):
    """ Test accuracy of storage. """
    M = simulation.StateMonitor(self.G, ['v', 'a', 'b'])
    sim = simulation.Simulation(self.G, M, dt=self.dt)
    sim.run(self.t_max)

    v_expected = np.array([i*(M.t+self.dt) for i in xrange(self.N)])
    a_expected = np.array([(M.t+self.dt) for i in xrange(2)])
    b_expected = [((i + np.round(M.t/sim.dt)).astype(int) % 3) == 0
      for i in xrange(self.N)]

    self.assertTrue(np.allclose(M.v, v_expected))
    self.assertTrue(np.allclose(M.a, a_expected))
    self.assertTrue(np.allclose(M.b, b_expected))

  def test_accurate_interval(self):
    """ Test that storage is accurate with custom interval. """
    interval = 0.5
    M = simulation.StateMonitor(self.G, 'v', interval=interval)
    sim = simulation.Simulation(self.G, M, dt=self.dt)
    sim.run(self.t_max)

    v_expected = np.array([i*(M.t + self.dt) for i in xrange(self.N)])
    self.assertTrue(np.allclose(M.v, v_expected))

####################
# TestTableSpikers #
####################

class TestTableSpikers(unittest.TestCase):
  def test_correct_spiking(self):
    """ Test that spiking happens when it should. """
    n = 10
    t_max = 25.0
    dt = 0.2
    p = 0.05

    # some reproducible arbitrariness
    np.random.seed(622312)
    n_steps = int_r(t_max/dt)
    table = np.random.rand(n_steps, n) < p

    G = TableSpikers(n)
    G.spike_table = copy.copy(table)

    class SimpleMonitor(object):
      def __init__(self, target):
        self.target = target;
        self.results = []
        self.order = 1

      def evolve(self, t, dt):
        idxs = self.target.spike.nonzero()[0]
        self.results.extend([(int_r(t/dt), i) for i in idxs])

    M = SimpleMonitor(G)
    sim = simulation.Simulation(G, M, dt=dt)
    sim.run(t_max)

    expected = zip(*table.nonzero())

    self.assertSequenceEqual(expected, M.results)

  def test_no_spike_after_table(self):
    """ Test that there are no more spikes past the end of the table. """
    n = 5
    dt = 1.0
    t_max = 2*dt
    # make sure we have spikes at the end
    table = np.ones((1, n))

    G = TableSpikers(n)
    G.spike_table = table

    sim = simulation.Simulation(G, dt=dt)
    sim.run(t_max)
    
    self.assertFalse(np.any(G.spike))

  def test_out(self):
    """ Test generation of output field. """
    t_max = 24.0
    dt = 0.1
    n_steps = int_r(t_max/dt)

    spike_t = 5.0
    spike_n = int_r(spike_t/dt)

    G = TableSpikers(1)

    table = np.zeros((n_steps, 1))
    table[spike_n, 0] = True
    G.spike_table = table

    Mo = simulation.StateMonitor(G, 'out')
    sim = simulation.Simulation(G, Mo, dt=dt)
    sim.run(t_max)

    mask = (Mo.t > spike_t)

    out_t = Mo.t[mask]
    out_y = Mo.out[0, mask]

    # there is a timing shift here between the actual output and the "expected"
    # one; I don't think this is an issue
    expected = out_y[0]*np.power(1 - dt/G.tau_out, (out_t - spike_t)/dt - 1)

    self.assertLess(np.mean(np.abs(out_y - expected)), 1e-6)

####################
# TestHVCLikeLayer #
####################

class TestHVCLikeLayer(unittest.TestCase):
  def test_jitter(self):
    """ Test that there are differences in spiking between trials. """
    # some reproducible arbitrariness
    np.random.seed(343143)

    n = 10
    t_max = 25
    dt = 0.1
    G = HVCLikeLayer(n)

    M1 = simulation.EventMonitor(G)

    sim1 = simulation.Simulation(G, M1, dt=dt)
    sim1.run(t_max)
    
    M2 = simulation.EventMonitor(G)
    sim2 = simulation.Simulation(G, M2, dt=dt)
    sim2.run(t_max)

    self.assertNotEqual(M1.t, M2.t)

  def test_no_jitter(self):
    """ Test that repeated noiseless trials are identical. """
    # some reproducible arbitrariness
    np.random.seed(3249823)

    n = 10
    t_max = 25
    dt = 0.1
    G = HVCLikeLayer(n)
    G.burst_noise = 0.0
    G.spike_noise = 0.0

    M1 = simulation.EventMonitor(G)

    sim1 = simulation.Simulation(G, M1, dt=dt)
    sim1.run(t_max)
    
    M2 = simulation.EventMonitor(G)
    sim2 = simulation.Simulation(G, M2, dt=dt)
    sim2.run(t_max)

    self.assertEqual(M1.t, M2.t)

  def test_uniform(self):
    """ Test that there are spikes all along the simulation window. """
    # some reproducible arbitrariness
    np.random.seed(87548)

    n = 50
    t_max = 50
    dt = 0.1
    resolution = 1.0

    class UniformityChecker(object):
      def __init__(self, target, resolution):
        self.target = target
        self.resolution = resolution
        self.order = 1

      def prepare(self, t_max, dt):
        self.has_spike = np.zeros(int_r(t_max/self.resolution) + 1)

      def evolve(self, t, dt):
        i = int_r(t/self.resolution)
        self.has_spike[i] = (self.has_spike[i] or np.any(self.target.spike))

    G = HVCLikeLayer(n)
    M = UniformityChecker(G, resolution)
    sim = simulation.Simulation(G, M, dt=dt)
    sim.run(t_max)

    self.assertTrue(np.all(M.has_spike))

  def test_burst(self):
    """ Test that each neuron fires a burst of given width and n_spikes. """
    n = 25
    t_max = 50
    dt = 0.1

    G = HVCLikeLayer(n)
    G.burst_noise = 0.0
    G.spike_noise = 0.0

    M = simulation.EventMonitor(G)
    sim = simulation.Simulation(G, M, dt=dt)
    sim.run(t_max)

    # split spikes by neuron index
    spikes = [np.asarray(M.t)[np.asarray(M.i) == i] for i in xrange(n)]

    self.assertTrue(np.all(len(_) == G.spikes_per_burst) for _ in spikes)

    burst_lengths = [_[-1] - _[0] for _ in spikes]
    self.assertLess(np.std(burst_lengths), dt/2)
    self.assertLess(np.abs(np.mean(burst_lengths) - 1000*(G.spikes_per_burst-1)
        / G.rate_during_burst), dt/2)

  def test_firing_rate_during_burst(self):
    # some reproducible arbitrariness
    np.random.seed(43245)

    n = 25
    t_max = 50
    dt = 0.1
    
    G = HVCLikeLayer(n)
    M = simulation.EventMonitor(G)
    sim = simulation.Simulation(G, M, dt=dt)
    sim.run(t_max)

    # split spikes by neuron index
    spikes = [np.asarray(M.t)[np.asarray(M.i) == i] for i in xrange(n)]

    # check that inter-spike intervals are in the correct range
    isi = [np.diff(_) for _ in spikes]
    isi_max = [np.max(_) for _ in isi]
    isi_min = [np.max(_) for _ in isi]
    spike_dt = 1000.0/G.rate_during_burst
    self.assertLess(np.max(isi_max), spike_dt + G.spike_noise + dt/2)
    self.assertGreater(np.min(isi_min), spike_dt - G.spike_noise - dt/2)

  def test_burst_dispersion(self):
    """ Test that starting times of bursts are within required bounds. """
    # some reproducible arbitrariness
    np.random.seed(7342642)

    n = 25
    t_max = 50
    dt = 0.1
    n_sim = 10
    
    G = HVCLikeLayer(n)

    burst_starts = []
    for i in xrange(n_sim):
      M = simulation.EventMonitor(G)
      sim = simulation.Simulation(G, M, dt=dt)
      sim.run(t_max)

      # split spikes by neuron index
      spikes = [np.asarray(M.t)[np.asarray(M.i) == i] for i in xrange(n)]
      burst_starts.append([_[0] for _ in spikes])

    burst_starts_range = [np.ptp([_[i] for _ in burst_starts])
        for i in xrange(n)]
    
    self.assertLess(np.max(burst_starts_range), G.burst_noise + dt/2)

  def test_burst_tmax(self):
    """ Test using a different end time for bursts than for simulation. """
    n = 25
    t_max = 50
    dt = 0.1

    G = HVCLikeLayer(n)
    G.burst_noise = 0.0
    G.spike_noise = 0.0

    M1 = simulation.EventMonitor(G)
    sim1 = simulation.Simulation(G, M1, dt=dt)
    sim1.run(t_max)

    G = HVCLikeLayer(n, burst_tmax=50)
    G.burst_noise = 0.0
    G.spike_noise = 0.0

    M2 = simulation.EventMonitor(G)
    sim2 = simulation.Simulation(G, M2, dt=dt)
    sim2.run(2*t_max)

    self.assertTrue(np.allclose(M1.t, M2.t))
    self.assertTrue(np.allclose(M1.i, M2.i))


####################
# TestRandomLayer  #
####################

class TestRandomLayer(unittest.TestCase):
  def test_variability(self):
    """ Test that output is variable. """
    # some reproducible arbitrariness
    np.random.seed(343143)

    n = 10
    t_max = 20.0
    dt = 0.1
    G = RandomLayer(n)

    M1 = simulation.EventMonitor(G)

    sim1 = simulation.Simulation(G, M1, dt=dt)
    sim1.run(t_max)
    
    M2 = simulation.EventMonitor(G)
    sim2 = simulation.Simulation(G, M2, dt=dt)
    sim2.run(t_max)

    self.assertNotEqual(len(M1.t), 0)
    self.assertNotEqual(len(M2.t), 0)
    self.assertNotEqual(M1.t, M2.t)

  def test_init_out_with_rate(self):  
    """ Test that initial value of `out` is given by `rate`. """
    n = 3
    rates = [10, 50, 90]

    G = RandomLayer(n, ini_rate=rates)
    G.prepare(10.0, 0.1)

    self.assertLess(np.max(np.abs(G.out - rates)), 1e-9)

####################
# TestStudentLayer #
####################

class TestStudentLayer(unittest.TestCase):
  def test_dynamics_no_tau_ref(self):
    """ Test dynamics with no refractory period. """
    n = 50
    t_max = 100.0
    dt = 0.1

    G = StudentLayer(n)
    G.tau_ref = 0.0

    i_values = np.linspace(0.01, 0.4, 50)

    for i_ext in i_values:
      # start with different initial voltages to take advantage of averaging
      # effects
      G.v_init = np.linspace(G.vR, G.v_th, n, endpoint=False)
      G.i_ext_init = i_ext

      M = simulation.EventMonitor(G)

      sim = simulation.Simulation(G, M, dt=dt)
      sim.run(t_max)
      
      rate = float(len(M.t))/n/t_max*1000.0
      # first source of uncertainty: a spike might not fit before the end of a
      # simulation
      uncertainty1 = 1.0/np.sqrt(n)/t_max*1000.0
      
      expected = 0.0
      uncertainty = uncertainty1
      if G.R*i_ext > G.v_th - G.vR:
        expected = 1000.0/(G.tau_m*np.log(G.R*i_ext/(G.vR-G.v_th+G.R*i_ext)))
        # second source of uncertainty: spikes might move due to the granularity
        # of the simulation
        uncertainty2 = dt*expected*rate/1000.0
        uncertainty = uncertainty1 + uncertainty2
        uncertainty *= 1.5
        self.assertLess(np.abs(rate - expected), uncertainty)
      else:
        self.assertAlmostEqual(rate, 0.0)

  def test_dynamics_with_tau_ref(self):
    """ Test dynamics with refractory period. """
    n = 10
    t_max = 100.0
    dt = 0.1

    G = StudentLayer(n)

    i_values = np.linspace(0.02, 0.4, 28)

    different = 0
    for i_ext in i_values:
      # start with different initial voltages to take advantage of averaging
      # effects
      G.v_init = np.linspace(G.vR, G.v_th, n, endpoint=False)
      G.i_ext_init = i_ext

      M = simulation.EventMonitor(G)

      sim = simulation.Simulation(G, M, dt=dt)
      sim.run(t_max)

      rate = float(len(M.t))/n/t_max*1000.0
      # first source of uncertainty: a spike might not fit before the end of a
      # simulation
      uncertainty1 = 1.0/np.sqrt(n)/t_max*1000.0
      
      expected0 = 0.0
      expected = 0.0
      if G.R*i_ext > G.v_th - G.vR:
        expected0 = 1000.0/(G.tau_m*np.log(G.R*i_ext/(G.vR-G.v_th+G.R*i_ext)))
        expected = expected0/(1 + expected0*G.tau_ref/1000.0)

        # second source of uncertainty: spikes might move due to the granularity
        # of the simulation
        uncertainty2 = dt*expected*rate/1000.0
        uncertainty = uncertainty1 + uncertainty2

        self.assertLess(np.abs(rate - expected), uncertainty)

        if np.abs(expected - expected0) >= uncertainty:
          different += 1
      else:
        self.assertAlmostEqual(rate, 0.0)
    
    # make sure that in most cases the firing rate using the refractory period
    # was significantly different from the case without refractory period
    self.assertGreater(different, len(i_values)*2/3)

  def test_v_bounds(self):
    """ Test that the membrane potential stays below threshold potential. """
    n = 50
    t_max = 100.0
    dt = 0.1

    G = StudentLayer(n)
    G.i_ext_init = np.linspace(-1.0, 1.0, n)

    class BoundsChecker(object):
      def __init__(self, target):
        self.target = target
        self.small = None
        self.large = None
        self.order = 1

      def evolve(self, t, dt):
        small = np.min(self.target.v)
        large = np.max(self.target.v)
        if self.small is None or self.small > small:
          self.small = small
        if self.large is None or self.large < large:
          self.large = large
    
    M = BoundsChecker(G)

    sim = simulation.Simulation(G, M, dt=dt)
    sim.run(t_max)

    self.assertLess(M.large, G.v_th)

  def test_out(self):
    """ Test generation of output field. """
    t_max = 24.0
    dt = 0.1

    G = StudentLayer(1)
    G.i_ext_init = 0.1

    M = simulation.EventMonitor(G)
    Mo = simulation.StateMonitor(G, 'out')
    sim = simulation.Simulation(G, M, Mo, dt=dt)
    sim.run(t_max)

    # we need a single spike for this
    # XXX we could also have set the refractory period to a really high number
    self.assertEqual(len(M.t), 1)

    t_spike = M.t[0]
    mask = (Mo.t > t_spike)

    out_t = Mo.t[mask]
    out_y = Mo.out[0, mask]

    expected = out_y[0]*np.power(1 - dt/G.tau_out, (out_t - t_spike)/dt)

    self.assertLess(np.mean(np.abs(out_y - expected)), 1e-6)

##########################
# TestExcitatorySynapses #
##########################

class TestExcitatorySynapses(unittest.TestCase):
  def setUp(self):
    # generate pseudo-random test case
    np.random.seed(123456)

    self.t_max = 16.0  # duration of simulation
    self.dt = 1.0     # time step
    self.N = 15       # number of units in source layer
    self.M = 30       # number of units in target layer
    self.p = 0.2      # probability of spiking per time step

    self.G = TableSpikers(self.N)
    self.G.spike_table = (np.random.rand(int_r(self.t_max/self.dt), self.N) <
        self.p)

    # a simple target layer
    class TargetNeurons(object):
      def __init__(self, N, v_step=1.0):
        self.N = N
        self.v_step = v_step
        self.active_state = True

      def prepare(self, t_max, dt):
        self.v = np.zeros(self.N)
        self.i_ampa = np.zeros(self.N)
        self.i_nmda = np.zeros(self.N)
        self.active = np.repeat(self.active_state, self.N)

      def evolve(self, t, dt):
        self.v += self.v_step

    self.Gp = TargetNeurons(self.N, np.inf)
    self.T = TargetNeurons(self.M, np.inf)

    self.syn_1t1 = ExcitatorySynapses(self.G, self.Gp)
    self.syn_dense = ExcitatorySynapses(self.G, self.T)

  def test_one_to_one_mismatch(self):
    """ Test exception for 1-to-1 synapses b/w layers of different sizes. """
    self.assertRaises(Exception, ExcitatorySynapses, self.G, self.T,
        one_to_one=True)

  def test_one_to_one_transmission_ampa(self):
    self.syn_1t1.W = np.linspace(0.1, 2.0, self.N)

    sim = simulation.Simulation(self.G, self.Gp, self.syn_1t1, dt=self.dt)
    sim.run(self.t_max)

    expected = np.zeros(self.N)
    for i in xrange(len(self.G.spike_table)):
      expected += self.syn_1t1.W*self.G.spike_table[i]

    self.assertTrue(np.allclose(expected, self.Gp.i_ampa))
    self.assertAlmostEqual(np.linalg.norm(self.Gp.i_nmda), 0.0)

  def test_one_to_one_transmission_nmda(self):
    self.syn_1t1.W = np.asarray([_ % 2 for _ in xrange(self.N)])
    self.syn_1t1.f_nmda = 1.0

    sim = simulation.Simulation(self.G, self.Gp, self.syn_1t1, dt=self.dt)
    sim.run(self.t_max)

    self.assertAlmostEqual(np.linalg.norm(self.Gp.i_ampa), 0.0)
    self.assertAlmostEqual(np.linalg.norm(self.Gp.i_nmda[::2].ravel()), 0.0)

    expected = np.zeros(self.N)
    v = np.zeros(self.N)

    for i in xrange(len(self.G.spike_table)):
      v += self.Gp.v_step
      expected[1::2] += self.G.spike_table[i, 1::2]/(1.0 + self.syn_1t1.mg/3.57*
          np.exp(-v[1::2]/16.13))

    self.assertTrue(np.allclose(expected, self.Gp.i_nmda))

  def test_dense_transmission(self):
    """ Test transmission with one-to-one synapses. """
    # generate pseudo-random test case
    f = 0.5

    np.random.seed(6564)
    self.syn_dense.W = np.random.randn(self.M, self.N)
    self.syn_dense.f_nmda = f

    sim = simulation.Simulation(self.G, self.T, self.syn_dense, dt=self.dt)
    sim.run(self.t_max)

    expected_ampa = np.zeros(self.M)
    expected_nmda = np.zeros(self.M)
    v = np.zeros(self.M)

    for i in xrange(len(self.G.spike_table)):
      v += self.T.v_step
      effect = np.dot(self.syn_dense.W, self.G.spike_table[i])
      expected_ampa += effect
      expected_nmda += effect/(1.0 + self.syn_1t1.mg/3.57*
          np.exp(-v/16.13))

    self.assertTrue(np.allclose((1-f)*expected_ampa, self.T.i_ampa))
    self.assertTrue(np.allclose(f*expected_nmda, self.T.i_nmda))

  def test_no_effect_during_refractory(self):
    """ Check that there is no effect during refractory periods. """
    np.random.seed(6564)
    f = 0.5
    self.syn_dense.W = np.random.randn(self.M, self.N)
    self.syn_dense.f_nmda = f
    self.syn_dense.change_during_ref = False

    self.T.active_state = False

    sim = simulation.Simulation(self.G, self.T, self.syn_dense, dt=self.dt)
    sim.run(self.t_max)

    self.assertAlmostEqual(np.linalg.norm(self.T.i_ampa), 0.0)
    self.assertAlmostEqual(np.linalg.norm(self.T.i_nmda), 0.0)

  def test_allow_effect_during_refractory(self):
    """ Check that it's possible to have effect during refractory periods. """
    np.random.seed(6564)
    f = 0.5
    self.syn_dense.W = np.random.randn(self.M, self.N)
    self.syn_dense.f_nmda = f
    self.syn_dense.change_during_ref = True

    self.T.active_state = False

    sim = simulation.Simulation(self.G, self.T, self.syn_dense, dt=self.dt)
    sim.run(self.t_max)

    self.assertGreater(np.linalg.norm(self.T.i_ampa), 0.1)
    self.assertGreater(np.linalg.norm(self.T.i_nmda), 0.1)

  def test_init_i_ampa(self):
    self.syn_dense.W = np.ones((self.M, self.N))
    self.syn_1t1.W = np.ones(self.N)

    self.G.avg_rates = 1.0

    self.Gp.tau_ampa = 5.0
    self.T.tau_ampa = 5.0

    sim = simulation.Simulation(self.G, self.Gp, self.T,
        self.syn_dense, self.syn_1t1,
        dt=self.dt)

    sim.run(0)

    self.assertGreater(np.linalg.norm(self.T.i_ampa), 1e-3)
    self.assertGreater(np.linalg.norm(self.Gp.i_ampa), 1e-3)

  def test_init_i_nmda(self):
    self.syn_dense.W = np.ones((self.M, self.N))
    self.syn_1t1.W = np.ones(self.N)

    self.syn_dense.f_nmda = 1.0
    self.syn_1t1.f_nmda = 1.0

    self.G.avg_rates = 1.0

    self.Gp.tau_nmda = 100.0
    self.T.tau_nmda = 100.0

    sim = simulation.Simulation(self.G, self.Gp, self.T,
        self.syn_dense, self.syn_1t1,
        dt=self.dt)

    sim.run(0)

    self.assertGreater(np.linalg.norm(self.T.i_nmda), 1e-3)
    self.assertGreater(np.linalg.norm(self.Gp.i_nmda), 1e-3)


########################
# TestLinearController #
########################

class TestLinearController(unittest.TestCase):
  
  def setUp(self):
    self.dt = 1.0       # time step
    self.N = 24         # number of units in source layer

    out_step = 0.1 # amount by which `out` grows at each step

    self.G = SimpleNeurons(self.N)

  def test_zero(self):
    """ Test controller with vanishing weights. """
    controller = LinearController(self.G, 2, mode='zero')
    sim = simulation.Simulation(self.G, controller, dt=self.dt)
    sim.run(self.dt)

    self.assertAlmostEqual(np.linalg.norm(controller.W.ravel()), 0.0)
    self.assertAlmostEqual(np.linalg.norm(controller.out), 0.0)

  def test_sum(self):
    """ Test additive controller. """
    controller = LinearController(self.G, 2, mode='sum')
    self.G.out_fct = lambda _: np.hstack(((self.N/2)*[1], (self.N/2)*[-1]))
    sim = simulation.Simulation(self.G, controller, dt=self.dt)
    sim.run(self.dt)

    value = 1.0*self.dt/controller.tau
    self.assertTrue(np.allclose(controller.out, [value, -value]))

  def test_push_pull(self):
    """ Test push/pull controller. """
    controller = LinearController(self.G, 3, mode='pushpull')
    self.G.out_fct = lambda _: np.hstack((
        (self.N/6)*[1],
        (self.N/6)*[-1],
        (self.N/6)*[1],
        (self.N/6)*[1],
        (self.N/6)*[0],
        (self.N/6)*[0],
      ))
    sim = simulation.Simulation(self.G, controller, dt=self.dt)
    sim.run(self.dt)

    value = 1.0/2*self.dt/controller.tau
    self.assertTrue(np.allclose(controller.out, [2*value, 0, 0]))

  def test_bias_initial(self):
    """ Test that initial values start at bias. """
    biases = [1, -1]

    controller = LinearController(self.G, 2, mode='zero')
    controller.bias = biases

    sim = simulation.Simulation(self.G, controller, dt=self.dt)
    sim.run(0)

    self.assertTrue(np.allclose(controller.out, biases))

  def test_bias(self):
    """ Test controller bias. """
    biases = [1, -0.5, 0.5, 1.5]

    controller = LinearController(self.G, 4, mode='zero')
    controller.bias = biases

    sim = simulation.Simulation(self.G, controller, dt=self.dt)
    sim.run(self.dt)

    self.assertTrue(np.allclose(controller.out, biases))

  def test_timescale(self):
    """ Test smoothing timescale. """
    tau = 25.0
    tmax = 50.0

    controller = LinearController(self.G, 1, mode='sum', tau=tau)
    self.G.out_fct = lambda _: np.ones(self.N)
    sim = simulation.Simulation(self.G, controller, dt=self.dt)
    sim.run(tmax)

    expected = 1.0 - (1.0 - self.dt/tau)**int_r(tmax/self.dt)
    self.assertTrue(np.allclose(controller.out, expected))

  def test_no_smoothing(self):
    """ Test the controller without smoothing. """
    # reproducible arbitrariness
    np.random.seed(12321)

    nsteps = 10
    tmax = nsteps*self.dt
    sequence = np.random.randn(nsteps)

    controller = LinearController(self.G, 1, mode='sum', tau=None)
    M = simulation.StateMonitor(controller, 'out')
    self.G.out_fct = lambda i: sequence[i]*np.ones(self.N)
    sim = simulation.Simulation(self.G, controller, M, dt=self.dt)
    sim.run(tmax)

    for i in xrange(nsteps):
      self.assertTrue(np.allclose(M.out[:, i], sequence[i]))

  def test_source_error(self):
    """ Test calculation of motor error mapped to source neurons. """
    # reproducible arbitrariness
    np.random.seed(12321)

    nsteps = 10
    nchan = 3
    tmax = nsteps*self.dt
    sequence = np.random.randn(nsteps, self.N)

    target = np.random.randn(nchan, nsteps)
    controller = LinearController(self.G, target, tau=None)
    controller.W = np.random.randn(*controller.W.shape)

    self.G.out_fct = lambda i: sequence[i]

    class SourceErrorGrabber(object):
      def __init__(self, target):
        self.target = target
        self.order = 10
      
      def prepare(self, tmax, dt):
        nsteps = int_r(tmax/dt)
        self.motor_error = np.zeros((nsteps, self.target.source.N))

      def evolve(self, t, dt):
        i = int_r(t/dt)
        self.motor_error[i, :] = self.target.get_source_error()

    M = SourceErrorGrabber(controller)
    M1 = simulation.StateMonitor(controller, 'out')

    sim = simulation.Simulation(self.G, controller, M, M1, dt=self.dt)
    sim.run(tmax)

    for i in xrange(int_r(tmax/self.dt)):
      diff = M1.out[:, i] - target[:, i]
      self.assertTrue(np.allclose(M.motor_error[i],
          np.dot(diff, controller.W)))

  def test_motor_error(self):
    """ Test calculation of motor error. """
    # reproducible arbitrariness
    np.random.seed(12325)

    nsteps = 10
    nchan = 3
    tmax = nsteps*self.dt
    sequence = np.random.randn(nsteps, self.N)

    target = np.random.randn(nchan, nsteps)
    controller = LinearController(self.G, target, tau=None)
    controller.W = np.random.randn(*controller.W.shape)

    self.G.out_fct = lambda i: sequence[i]

    class MotorErrorGrabber(object):
      def __init__(self, target):
        self.target = target
        self.order = 10
      
      def prepare(self, tmax, dt):
        nsteps = int_r(tmax/dt)
        self.motor_error = np.zeros((nsteps, self.target.N))

      def evolve(self, t, dt):
        i = int_r(t/dt)
        self.motor_error[i, :] = self.target.get_motor_error()

    M = MotorErrorGrabber(controller)
    M1 = simulation.StateMonitor(controller, 'out')

    sim = simulation.Simulation(self.G, controller, M, M1, dt=self.dt)
    sim.run(tmax)

    for i in xrange(int_r(tmax/self.dt)):
      diff = M1.out[:, i] - target[:, i]
      self.assertTrue(np.allclose(M.motor_error[i], diff))

  def test_permute_inverse(self):
    """ Test that `permute_inverse` works. """
    # reproducible arbitrariness
    np.random.seed(12321)

    nsteps = 20
    nchan = 3
    tmax = nsteps*self.dt
    sequence = np.random.randn(nsteps, self.N)

    permutation = np.arange(self.N)
    n1a = 3
    n1b = 5
    n2a = 13
    n2b = 4
    permutation[n1a], permutation[n1b] = (permutation[n1b], permutation[n1a])
    permutation[n2a], permutation[n2b] = (permutation[n2b], permutation[n2a])

    target = np.random.randn(nchan, nsteps)
    controller = LinearController(self.G, target, tau=None)
    controller.W = np.random.randn(*controller.W.shape)

    self.G.out_fct = lambda i: sequence[i]

    class SourceErrorGrabber(object):
      def __init__(self, target):
        self.target = target
        self.order = 10
      
      def prepare(self, tmax, dt):
        nsteps = int_r(tmax/dt)
        self.motor_error = np.zeros((nsteps, self.target.source.N))

      def evolve(self, t, dt):
        i = int_r(t/dt)
        self.motor_error[i, :] = self.target.get_source_error()

    ME1 = SourceErrorGrabber(controller)

    sim1 = simulation.Simulation(self.G, controller, ME1, dt=self.dt)
    sim1.run(tmax)

    controller.permute_inverse = permutation
    ME2 = SourceErrorGrabber(controller)

    sim2 = simulation.Simulation(self.G, controller, ME2, dt=self.dt)
    sim2.run(tmax)

    # test that the correct source error outputs have been swapped
    expected = np.copy(ME1.motor_error)
    expected[:, [n1a, n1b]] = expected[:, [n1b, n1a]]
    expected[:, [n2a, n2b]] = expected[:, [n2b, n2a]]

    self.assertAlmostEqual(np.mean(np.abs(expected - ME2.motor_error)), 0.0)

  def test_random_permute_inverse_fraction(self):
    """ Test random permutation shuffles correct fraction of neurons. """
    # reproducible arbitrariness
    np.random.seed(12325)

    nchan = 3
    nsteps = 20
    rho = 1.0/4
    target = np.random.randn(nchan, nsteps)

    controller = LinearController(self.G, target, tau=None)

    controller.set_random_permute_inverse(rho)
    self.assertIsNotNone(controller.permute_inverse)

    # check that the right fraction of assignments are kept intact
    self.assertEqual(np.sum(controller.permute_inverse == np.arange(self.N)),
        (1.0 - rho)*self.N)

  def test_random_permute_inverse_changes_group(self):
    """ Test random permutation moves affected neurons to different groups. """
    # reproducible arbitrariness
    np.random.seed(232)

    nchan = 3
    nsteps = 20
    rho = 1.0/4
    target = np.random.randn(nchan, nsteps)

    controller = LinearController(self.G, target, tau=None)

    controller.set_random_permute_inverse(rho)
    self.assertIsNotNone(controller.permute_inverse)

    n_per_group = self.N/nchan
    groups0 = np.arange(self.N)/n_per_group
    groups1 = controller.permute_inverse/n_per_group

    # check that the right fraction of assignments are kept intact
    self.assertEqual(np.sum(groups0 != groups1), rho*self.N)

  def test_random_permute_inverse_is_random(self):
    """ Test random permutation moves changes between trials. """
    # reproducible arbitrariness
    np.random.seed(2325)

    nchan = 3
    nsteps = 20
    rho = 1.0/4
    target = np.random.randn(nchan, nsteps)

    controller = LinearController(self.G, target, tau=None)

    controller.set_random_permute_inverse(rho)
    self.assertIsNotNone(controller.permute_inverse)

    perm1 = np.copy(controller.permute_inverse)

    controller.set_random_permute_inverse(rho)
    perm2 = controller.permute_inverse

    self.assertNotEqual(np.sum(perm1 == perm2), self.N)

  def test_random_permute_inverse_subdivide(self):
    """ Test `subdivid_by` option for random permutation. """
    # reproducible arbitrariness
    np.random.seed(121)

    nchan = 3
    nsteps = 20
    rho = 1.0/2
    subdiv = 2
    target = np.random.randn(nchan, nsteps)

    controller = LinearController(self.G, target, tau=None)

    controller.set_random_permute_inverse(rho, subdivide_by=subdiv)
    self.assertIsNotNone(controller.permute_inverse)

    n_per_group = self.N/nchan
    groups0 = np.arange(self.N)/n_per_group
    groups1 = controller.permute_inverse/n_per_group

    n_per_subgroup = self.N/(subdiv*nchan)
    subgroups0 = np.arange(self.N)/n_per_subgroup
    subgroups1 = controller.permute_inverse/n_per_subgroup

    # check that the right fraction of assignments are kept intact
    self.assertEqual(np.sum(subgroups0 != subgroups1), rho*self.N)
    
    # but that some of the mismatches end up *within the same group*
    # (though they come from different subgroups)
    self.assertNotEqual(np.sum(groups0 != groups1), rho*self.N)

  def test_error_map_fct(self):
    """ Test mapping of the source error through a nonlinearity. """
    # reproducible arbitrariness
    np.random.seed(2343)

    nsteps = 12
    nchan = 4
    tmax = nsteps*self.dt
    sequence = np.random.randn(nsteps, self.N)

    target = np.random.randn(nchan, nsteps)
    controller = LinearController(self.G, target, tau=None)
    controller.W = np.random.randn(*controller.W.shape)
    controller.error_map_fct = lambda err: np.tanh(err)

    self.G.out_fct = lambda i: sequence[i]

    class SourceErrorGrabber(object):
      def __init__(self, target):
        self.target = target
        self.order = 10
      
      def prepare(self, tmax, dt):
        nsteps = int_r(tmax/dt)
        self.motor_error = np.zeros((nsteps, self.target.source.N))

      def evolve(self, t, dt):
        i = int_r(t/dt)
        self.motor_error[i, :] = self.target.get_source_error()

    M = SourceErrorGrabber(controller)
    M1 = simulation.StateMonitor(controller, 'out')

    sim = simulation.Simulation(self.G, controller, M, M1, dt=self.dt)
    sim.run(tmax)

    for i in xrange(int_r(tmax/self.dt)):
      diff = M1.out[:, i] - target[:, i]
      self.assertTrue(np.allclose(M.motor_error[i],
          np.dot(controller.error_map_fct(diff), controller.W)))

  def test_nonlinearity(self):
    """ Test linear-nonlinear model. """
    # reproducible arbitrariness
    np.random.seed(1232321)

    nsteps = 10
    tmax = nsteps*self.dt
    sequence = np.random.randn(nsteps)

    controller = LinearController(self.G, 3, mode='sum', tau=20.0)

    M1 = simulation.StateMonitor(controller, 'out')
    self.G.out_fct = lambda i: sequence[i]*np.ones(self.N)

    sim1 = simulation.Simulation(self.G, controller, M1, dt=self.dt)
    sim1.run(tmax)

    controller.nonlinearity = lambda v: v**2 - v

    M2 = simulation.StateMonitor(controller, 'out')
    sim2 = simulation.Simulation(self.G, controller, M2, dt=self.dt)
    sim2.run(tmax)

    self.assertLess(np.max(np.abs(M2.out - controller.nonlinearity(M1.out))),
                    1e-9)

#################################
# TestTwoExponentialsPlasticity #
#################################


# the tests themselves
class TestTwoExponentialsPlasticity(unittest.TestCase):

  def setUp(self):
    # generate pseudo-random test case
    self.dt = 1.0       # time step
    self.Nc = 15        # number of units in conductor layer
    self.Ns = 30        # number of units in student layer

    # a do-nothing layer
    class MockNeurons(object):
      def __init__(self, N):
        self.N = N

      def prepare(self, t_max, dt):
        self.v = np.zeros(self.N)
        self.i_ampa = np.zeros(self.N)
        self.i_nmda = np.zeros(self.N)

      def evolve(self, t, dt):
        pass

    class SimpleSynapses(object):
      def __init__(self, source, target):
        self.source = source
        self.target = target
        self.W = np.zeros((self.target.N, self.source.N))
        self.order = 1

      def evolve(self, t, dt):
        self.target.out = np.dot(self.W, self.source.out)

    self.conductor = SimpleNeurons(self.Nc)
    self.student = MockNeurons(self.Ns)
    self.tutor = SimpleNeurons(self.Ns)

    # reproducible arbitrariness
    np.random.seed(3231)

    self.syns = SimpleSynapses(self.conductor, self.student)
    self.syns.W = np.random.rand(*self.syns.W.shape)

    self.rule = TwoExponentialsPlasticity(self.syns, self.tutor,
                                          constrain_positive=False,
                                          rate=1-6)

  def test_linear_in_cond(self):
    """ Test that weight change is linear in conductor output. """
    # reproducible arbitrariness
    np.random.seed(3232)

    cond_out = np.random.randn(self.Nc)
    alpha = 2.3

    self.conductor.out_step = np.copy(cond_out)
    self.tutor.out_step = self.rule.theta + 10*np.random.randn(self.Ns)

    W0 = np.copy(self.syns.W)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim.run(self.dt)

    change1 = self.syns.W - W0

    self.syns.W = np.copy(W0)
    self.conductor.out_step = alpha*cond_out
    sim.run(self.dt)

    change2 = self.syns.W - W0

    self.assertTrue(np.allclose(change2, alpha*change1))

  def test_linear_in_tut(self):
    """ Test that weight change is linear in tutor output. """
    # reproducible arbitrariness
    np.random.seed(5000)

    tut_out = np.random.randn(self.Ns)
    alpha = 0.7

    self.conductor.out_step = np.random.randn(self.Nc)
    self.tutor.out_fct = lambda _: self.rule.theta + tut_out

    W0 = np.copy(self.syns.W)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim.run(self.dt)

    change1 = self.syns.W - W0

    self.syns.W = np.copy(W0)
    self.tutor.out_fct = lambda _: self.rule.theta + alpha*tut_out
    sim.run(self.dt)

    change2 = self.syns.W - W0

    self.assertTrue(np.allclose(change2, alpha*change1))

  def test_linear_in_rate(self):
    """ Test that weight change is linear in learning rate. """
    # reproducible arbitrariness
    np.random.seed(4901)

    alpha = 1.2

    self.conductor.out_step = np.random.randn(self.Nc)
    self.tutor.out_step = np.random.randn(self.Ns)

    W0 = np.copy(self.syns.W)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim.run(self.dt)

    change1 = self.syns.W - W0

    self.syns.W = np.copy(W0)
    self.rule.rate *= alpha
    sim.run(self.dt)

    change2 = self.syns.W - W0

    self.assertTrue(np.allclose(change2, alpha*change1))

  def test_constrain_positive(self):
    """ Test that we can force the weights to stay positive. """
    # first run without constraints and make sure some weights become negative
    # NB: need to divide by 2 because the SimpleLayer's `evolve` gets called
    # before the plasticity rule's `evolve`, and so the tutor output becomes
    # *twice* `out_step`
    self.tutor.out_step = self.rule.theta/2 + np.hstack(( np.ones(self.Ns/2),
                                                         -np.ones(self.Ns/2)))
    self.conductor.out_step = np.ones(self.Nc)

    self.syns.W = np.zeros(self.syns.W.shape)
    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim.run(self.dt)

    self.assertGreater(np.sum(self.syns.W < 0), 0)

    # next run with the constraint and check that everything stays positive
    self.rule.constrain_positive = True
    self.syns.W = np.zeros(self.syns.W.shape)
    sim.run(self.dt)

    self.assertEqual(np.sum(self.syns.W < 0), 0)

  def test_prop_alpha(self):
    """ Test that synaptic change is linear in `alpha`. """
    # reproducible arbitrariness
    np.random.seed(5001)

    self.conductor.out_step = np.random.randn(self.Nc)
    self.tutor.out_step = np.random.randn(self.Ns)

    self.rule.alpha = 1.0
    self.rule.beta = 0.0

    tmax = 5*self.dt
    factor = 1.3

    W0 = np.copy(self.syns.W)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim.run(tmax)

    change1 = self.syns.W - W0

    self.syns.W = np.copy(W0)
    self.rule.alpha *= factor
    sim.run(tmax)

    change2 = self.syns.W - W0

    self.assertTrue(np.allclose(change2, factor*change1))

  def test_prop_beta(self):
    """ Test that synaptic change is linear in `beta`. """
    # reproducible arbitrariness
    np.random.seed(1321)

    self.rule.alpha = 0
    self.rule.beta = 0.5

    self.conductor.out_step = np.random.randn(self.Nc)
    self.tutor.out_step = np.random.randn(self.Ns)

    factor = 1.5
    tmax = 7*self.dt

    W0 = np.copy(self.syns.W)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim.run(tmax)

    change1 = self.syns.W - W0

    self.syns.W = np.copy(W0)
    self.rule.beta *= factor
    sim.run(tmax)

    change2 = self.syns.W - W0

    self.assertTrue(np.allclose(change2, factor*change1))

  def test_additive_alpha_beta(self):
    """ Test that alpha and beta components are additive. """
    np.random.seed(912838)

    param_pairs = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    tmax = 4*self.dt

    self.conductor.out_step = np.random.randn(self.Nc)
    self.tutor.out_step = np.random.randn(self.Ns)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    W0 = np.copy(self.syns.W)
    changes = []

    for params in param_pairs:
      self.rule.alpha = params[0]
      self.rule.beta = params[1]

      self.syns.W = np.copy(W0)
      sim.run(tmax)

      changes.append(self.syns.W - W0)

    self.assertTrue(np.allclose(changes[-1], changes[0] + changes[1]))

  def test_timescales(self):
    """ Test the timescales for alpha and beta components. """
    np.random.seed(2312321)
    param_pairs = [(1, 0, self.rule.tau1), (0, 1, self.rule.tau2)]

    nsteps = 10
    self.conductor.out_fct = lambda i: 10*np.ones(self.Nc) if i == 0 \
      else np.zeros(self.Nc)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    W0 = np.copy(self.syns.W)

    for params in param_pairs:
      self.rule.alpha = params[0]
      self.rule.beta = params[1]
      tau = params[2]

      self.tutor.out_fct = lambda i: (self.rule.theta + (10 if i == 0 else 0))*\
        np.ones(self.Ns)

      self.syns.W = np.copy(W0)
      sim.run(self.dt)

      change0 = self.syns.W - W0

      self.assertGreater(np.linalg.norm(change0), 1e-10)
      
      self.tutor.out_fct = lambda i: (self.rule.theta + (10
        if i == nsteps-1 else 0))*np.ones(self.Ns)

      self.syns.W = np.copy(W0)
      sim.run(nsteps*self.dt)

      change1 = self.syns.W - W0

      change1_exp = change0*(1 - float(self.dt)/tau)**(nsteps-1)

      self.assertTrue(np.allclose(change1, change1_exp),
        msg="Timescale not verified, alpha={}, beta={}.".format(*params[:2]))

  def test_tuple_synapses(self):
    """ Test using a tuple instead of synapses object. """
    # reproducible arbitrariness
    np.random.seed(5003)

    self.conductor.out_step = np.random.randn(self.Nc)
    self.tutor.out_step = np.random.randn(self.Ns)

    self.rule.alpha = 1.0
    self.rule.beta = 1.5

    tmax = 10*self.dt

    W0 = np.copy(self.syns.W)

    sim1 = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim1.run(tmax)

    final1 = np.copy(self.syns.W)

    self.syns.W = np.copy(W0)

    rule2 = TwoExponentialsPlasticity(
        (self.syns.source, self.syns.target, self.syns.W),
        self.tutor, constrain_positive=False, rate=1-6)
    rule2.alpha = 1.0
    rule2.beta = 1.5

    sim2 = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, rule2, dt=self.dt)
    sim2.run(tmax)

    final2 = np.copy(self.syns.W)

    self.assertTrue(np.allclose(final1, final2))


##################################
# TestSuperExponentialPlasticity #
##################################

class TestSuperExponentialPlasticity(unittest.TestCase):

  def setUp(self):
    # generate pseudo-random test case
    self.dt = 1.0       # time step
    self.Nc = 15        # number of units in conductor layer
    self.Ns = 30        # number of units in student layer

    # a do-nothing layer
    class MockNeurons(object):
      def __init__(self, N):
        self.N = N

      def prepare(self, t_max, dt):
        self.v = np.zeros(self.N)
        self.i_ampa = np.zeros(self.N)
        self.i_nmda = np.zeros(self.N)

      def evolve(self, t, dt):
        pass

    class SimpleSynapses(object):
      def __init__(self, source, target):
        self.source = source
        self.target = target
        self.W = np.zeros((self.target.N, self.source.N))
        self.order = 1

      def evolve(self, t, dt):
        self.target.out = np.dot(self.W, self.source.out)

    self.conductor = SimpleNeurons(self.Nc)
    self.student = MockNeurons(self.Ns)
    self.tutor = SimpleNeurons(self.Ns)

    # reproducible arbitrariness
    np.random.seed(3231)

    self.syns = SimpleSynapses(self.conductor, self.student)
    self.syns.W = np.random.rand(*self.syns.W.shape)

    self.rule = SuperExponentialPlasticity(self.syns, self.tutor,
                                           constrain_positive=False,
                                           rate=1-6)

  def test_linear_in_cond(self):
    """ Test that weight change is linear in conductor output. """
    # reproducible arbitrariness
    np.random.seed(3232)

    cond_out = np.random.randn(self.Nc)
    alpha = 2.3

    self.conductor.out_step = np.copy(cond_out)
    self.tutor.out_step = self.rule.theta + 10*np.random.randn(self.Ns)

    W0 = np.copy(self.syns.W)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim.run(self.dt)

    change1 = self.syns.W - W0

    self.syns.W = np.copy(W0)
    self.conductor.out_step = alpha*cond_out
    sim.run(self.dt)

    change2 = self.syns.W - W0

    self.assertTrue(np.allclose(change2, alpha*change1))

  def test_linear_in_tut(self):
    """ Test that weight change is linear in tutor output. """
    # reproducible arbitrariness
    np.random.seed(5000)

    tut_out = np.random.randn(self.Ns)
    alpha = 0.7

    self.conductor.out_step = np.random.randn(self.Nc)
    self.tutor.out_fct = lambda _: self.rule.theta + tut_out

    W0 = np.copy(self.syns.W)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim.run(self.dt)

    change1 = self.syns.W - W0

    self.syns.W = np.copy(W0)
    self.tutor.out_fct = lambda _: self.rule.theta + alpha*tut_out
    sim.run(self.dt)

    change2 = self.syns.W - W0

    self.assertTrue(np.allclose(change2, alpha*change1))

  def test_linear_in_rate(self):
    """ Test that weight change is linear in learning rate. """
    # reproducible arbitrariness
    np.random.seed(4901)

    alpha = 1.2

    self.conductor.out_step = np.random.randn(self.Nc)
    self.tutor.out_step = np.random.randn(self.Ns)

    W0 = np.copy(self.syns.W)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim.run(self.dt)

    change1 = self.syns.W - W0

    self.syns.W = np.copy(W0)
    self.rule.rate *= alpha
    sim.run(self.dt)

    change2 = self.syns.W - W0

    self.assertTrue(np.allclose(change2, alpha*change1))

  def test_constrain_positive(self):
    """ Test that we can force the weights to stay positive. """
    # first run without constraints and make sure some weights become negative
    # NB: need to divide by 2 because the SimpleLayer's `evolve` gets called
    # before the plasticity rule's `evolve`, and so the tutor output becomes
    # *twice* `out_step`
    self.tutor.out_step = self.rule.theta/2 + np.hstack(( np.ones(self.Ns/2),
                                                         -np.ones(self.Ns/2)))
    self.conductor.out_step = np.ones(self.Nc)

    self.syns.W = np.zeros(self.syns.W.shape)
    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim.run(self.dt)

    self.assertGreater(np.sum(self.syns.W < 0), 0)

    # next run with the constraint and check that everything stays positive
    self.rule.constrain_positive = True
    self.syns.W = np.zeros(self.syns.W.shape)
    sim.run(self.dt)

    self.assertEqual(np.sum(self.syns.W < 0), 0)

  def test_prop_alpha(self):
    """ Test that synaptic change is linear in `alpha`. """
    # reproducible arbitrariness
    np.random.seed(5001)

    self.conductor.out_step = np.random.randn(self.Nc)
    self.tutor.out_step = np.random.randn(self.Ns)

    self.rule.alpha = 1.0
    self.rule.beta = 0.0

    tmax = 5*self.dt
    factor = 1.3

    W0 = np.copy(self.syns.W)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim.run(tmax)

    change1 = self.syns.W - W0

    self.syns.W = np.copy(W0)
    self.rule.alpha *= factor
    sim.run(tmax)

    change2 = self.syns.W - W0

    self.assertTrue(np.allclose(change2, factor*change1))

  def test_prop_beta(self):
    """ Test that synaptic change is linear in `beta`. """
    # reproducible arbitrariness
    np.random.seed(1321)

    self.rule.alpha = 0
    self.rule.beta = 0.5

    self.conductor.out_step = np.random.randn(self.Nc)
    self.tutor.out_step = np.random.randn(self.Ns)

    factor = 1.5
    tmax = 7*self.dt

    W0 = np.copy(self.syns.W)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim.run(tmax)

    change1 = self.syns.W - W0

    self.syns.W = np.copy(W0)
    self.rule.beta *= factor
    sim.run(tmax)

    change2 = self.syns.W - W0

    self.assertTrue(np.allclose(change2, factor*change1))

  def test_additive_alpha_beta(self):
    """ Test that alpha and beta components are additive. """
    np.random.seed(912838)

    param_pairs = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    tmax = 4*self.dt

    self.conductor.out_step = np.random.randn(self.Nc)
    self.tutor.out_step = np.random.randn(self.Ns)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    W0 = np.copy(self.syns.W)
    changes = []

    for params in param_pairs:
      self.rule.alpha = params[0]
      self.rule.beta = params[1]

      self.syns.W = np.copy(W0)
      sim.run(tmax)

      changes.append(self.syns.W - W0)

    self.assertTrue(np.allclose(changes[-1], changes[0] + changes[1]))

  def test_timescale_beta(self):
    """ Test the timescale for beta component. """
    param_pairs = [(0, 1, self.rule.tau2)]

    nsteps = 10
    self.conductor.out_fct = lambda i: 10*np.ones(self.Nc) if i == 0 \
      else np.zeros(self.Nc)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    W0 = np.copy(self.syns.W)

    for params in param_pairs:
      self.rule.alpha = params[0]
      self.rule.beta = params[1]
      tau = params[2]

      self.tutor.out_fct = lambda i: (self.rule.theta + (10 if i == 0 else 0))*\
        np.ones(self.Ns)

      self.syns.W = np.copy(W0)
      sim.run(self.dt)

      change0 = self.syns.W - W0

      self.assertGreater(np.linalg.norm(change0), 1e-10)
      
      self.tutor.out_fct = lambda i: (self.rule.theta + (10
        if i == nsteps-1 else 0))*np.ones(self.Ns)

      self.syns.W = np.copy(W0)
      sim.run(nsteps*self.dt)

      change1 = self.syns.W - W0

      change1_exp = change0*(1 - float(self.dt)/tau)**(nsteps-1)

      self.assertTrue(np.allclose(change1, change1_exp),
        msg="Timescale not verified, alpha={}, beta={}.".format(*params[:2]))

  def test_super_exponential(self):
    """ Test alpha component goes like t*e^{-t}. """
    nsteps = 100
    self.dt = 0.1
    self.conductor.out_fct = lambda i: 10*np.ones(self.Nc) if i == 0 \
      else np.zeros(self.Nc)

    sim = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    W0 = np.copy(self.syns.W)

    self.rule.alpha = 1
    self.rule.beta = 0
    tau = self.rule.tau1

    j1 = nsteps/3
    j2 = nsteps

    self.tutor.out_fct = lambda i: (self.rule.theta +
        (10 if i == j1-1 else 0))*np.ones(self.Ns)
    delta1 = j1*self.dt

    self.syns.W = np.copy(W0)
    sim.run(delta1)

    change1 = self.syns.W - W0
    self.assertGreater(np.linalg.norm(change1), 1e-10)
      
    self.tutor.out_fct = lambda i: (self.rule.theta +
        (10 if i == j2-1 else 0))*np.ones(self.Ns)
    delta2 = j2*self.dt

    self.syns.W = np.copy(W0)
    sim.run(delta2)
    change2 = self.syns.W - W0
    self.assertGreater(np.linalg.norm(change2), 1e-10)

    ratio = change1/change2
    ratio_exp = ((delta1/delta2)*(np.exp(-(delta1 - delta2)/tau))
        *np.ones(np.shape(ratio)))

    self.assertLess(np.max(np.abs(ratio - ratio_exp)/ratio), 0.05)

  def test_tuple_synapses(self):
    """ Test using a tuple instead of synapses object. """
    # reproducible arbitrariness
    np.random.seed(5003)

    self.conductor.out_step = np.random.randn(self.Nc)
    self.tutor.out_step = np.random.randn(self.Ns)

    self.rule.alpha = 1.0
    self.rule.beta = 1.5

    tmax = 10*self.dt

    W0 = np.copy(self.syns.W)

    sim1 = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, self.rule, dt=self.dt)
    sim1.run(tmax)

    final1 = np.copy(self.syns.W)

    self.syns.W = np.copy(W0)

    rule2 = SuperExponentialPlasticity(
        (self.syns.source, self.syns.target, self.syns.W),
        self.tutor, constrain_positive=False, rate=1-6)
    rule2.alpha = 1.0
    rule2.beta = 1.5

    sim2 = simulation.Simulation(self.conductor, self.student, self.tutor,
                                 self.syns, rule2, dt=self.dt)
    sim2.run(tmax)

    final2 = np.copy(self.syns.W)

    self.assertTrue(np.allclose(final1, final2))


#########################
# TestBlackboxTutorRule #
#########################

class TestBlackboxTutorRule(unittest.TestCase):

  def setUp(self):
    self.Nsrc = 12      # number of source neurons
    self.Nout = 3       # number of output channels

    class MockSource(object):
      def __init__(self, N):
        self.N = N

      def evolve(self, t, dt):
        pass

    class MockController(object):
      def __init__(self, N, source, error_fct):
        self.N = N
        self.source = source
        self.error_fct = error_fct
        self.order = 2

      def prepare(self, tmax, dt):
        self._last_error = self.error_fct(0)

      def evolve(self, t, dt):
        self._last_error = self.error_fct(t)

      def get_source_error(self):
        return self._last_error

    self.source = MockSource(self.Nsrc)
    self.motor = MockController(self.Nout, self.source, lambda _: 0)
    self.rule = BlackboxTutorRule(self.motor, gain=1)

  def test_memory(self):
    """ Test the timescale of the integration. """
    tau = 53.0
    tau0 = 22.0
    mrate = 50.0
    Mrate = 100.0

    tmax = 100.0
    dt = 0.01

    self.rule.tau = tau
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False

    ndiv3 = self.Nsrc/3

    self.motor.error_fct = lambda t: np.hstack((
      np.cos(t/tau0)*np.ones(ndiv3), np.sin(t/tau0)*np.ones(ndiv3),
      np.ones(ndiv3)))

    M = simulation.StateMonitor(self.rule, 'out')

    sim = simulation.Simulation(self.source, self.motor, self.rule, M, dt=dt)
    sim.run(tmax)

    # tutor output points *opposite* the motor error!
    prefactor = -self.rule.gain*tau0/(tau*tau + tau0*tau0)
    integral_part1 = np.cos(M.t/tau0)*np.exp(-M.t/tau)
    integral_part2 = np.sin(M.t/tau0)*np.exp(-M.t/tau)

    expected_cos = prefactor*(tau0 - tau0*integral_part1 + tau*integral_part2)
    expected_sin = prefactor*(tau - tau*integral_part1 - tau0*integral_part2)
    expected_const = -(1 - np.exp(-M.t/tau))

    mavg = (mrate + Mrate)*0.5
    mdiff = (Mrate - mrate)*0.5
    expected = np.vstack((
        np.tile(mavg + mdiff*expected_cos, (ndiv3, 1)),
        np.tile(mavg + mdiff*expected_sin, (ndiv3, 1)),
        np.tile(mavg + mdiff*expected_const, (ndiv3, 1))
      ))

    # mismatch is relatively large since we're using Euler's method
    # we can't do much better, however, since the motor controller cannot give
    # us motor error information at sub-step resolution
    mismatch = np.mean(np.abs(expected - M.out)/expected)
    self.assertLess(mismatch, 0.05)

  def test_no_memory(self):
    """ Test instantaneous response. """
    tau0 = 23.0
    mrate = 50.0
    Mrate = 100.0

    tmax = 100.0
    dt = 0.2

    self.rule.tau = 0
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False

    ndiv3 = self.Nsrc/3

    self.motor.error_fct = lambda t: np.hstack((
      np.cos(t/tau0)*np.ones(ndiv3), np.sin(t/tau0)*np.ones(ndiv3),
      np.ones(ndiv3)))

    M = simulation.StateMonitor(self.rule, 'out')

    sim = simulation.Simulation(self.source, self.motor, self.rule, M, dt=dt)
    sim.run(tmax)

    mavg = (mrate + Mrate)*0.5
    mdiff = (Mrate - mrate)*0.5

    # tutor output points *opposite* the motor error!
    expected = mavg - mdiff*np.vstack((
      np.tile(np.cos(M.t/tau0), (ndiv3, 1)),
      np.tile(np.sin(M.t/tau0), (ndiv3, 1)),
      np.ones((ndiv3, len(M.t)))))
    
    mismatch = np.mean(np.abs(M.out - expected))

    self.assertAlmostEqual(mismatch, 0)

  def test_gain(self):
    """ Test gain controls. """
    tau = 50.0
    mrate = 50.0
    Mrate = 100.0
    gain = 5

    tmax = 50.0
    dt = 0.1

    self.rule.tau = tau
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False
    self.rule.gain = 1

    self.motor.error_fct = lambda _: np.ones(self.Nsrc)

    M1 = simulation.StateMonitor(self.rule, 'out')

    sim1 = simulation.Simulation(self.source, self.motor, self.rule, M1, dt=dt)
    sim1.run(tmax)

    self.rule.gain = gain

    M2 = simulation.StateMonitor(self.rule, 'out')

    sim2 = simulation.Simulation(self.source, self.motor, self.rule, M2, dt=dt)
    sim2.run(tmax)

    mavg = (mrate + Mrate)*0.5

    out1 = M1.out - mavg
    out2 = M2.out - mavg

    self.assertTrue(np.allclose(gain*out1, out2), msg=
      "mean(abs(gain*out1 - out2))={}".format(
        np.mean(np.abs(gain*out1 - out2))))

  def test_range_no_compress(self):
    """ Test range controls when there is no compression. """
    tau = 40.0
    mrate1 = 50.0
    Mrate1 = 100.0

    mrate2 = 30.0
    Mrate2 = 130.0

    tmax = 50.0
    dt = 0.1

    self.rule.tau = tau
    self.rule.min_rate = mrate1
    self.rule.max_rate = Mrate1
    self.rule.compress_rates = False

    self.motor.error_fct = lambda t: (int_r(t)%2)*np.ones(self.Nsrc)

    M1 = simulation.StateMonitor(self.rule, 'out')

    sim1 = simulation.Simulation(self.source, self.motor, self.rule, M1, dt=dt)
    sim1.run(tmax)

    self.rule.min_rate = mrate2
    self.rule.max_rate = Mrate2

    M2 = simulation.StateMonitor(self.rule, 'out')

    sim2 = simulation.Simulation(self.source, self.motor, self.rule, M2, dt=dt)
    sim2.run(tmax)

    expected2 = mrate2 + (M1.out - mrate1)*(Mrate2 - mrate2)/(Mrate1 - mrate1)

    self.assertTrue(np.allclose(M2.out, expected2), msg=
      "mean(abs(out2 - expected2))={}".format(
        np.mean(np.abs(M2.out - expected2))))

  def test_compress_works(self):
    """ Test that compression keeps firing rates from exceeding limits. """
    tau = 45.0
    mrate = 60.0
    Mrate = 100.0
    gain = 5

    tmax = 50.0
    dt = 0.2

    self.rule.tau = tau
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False
    self.rule.gain = gain

    self.motor.error_fct = lambda t: (int_r(t/20.0)%3-1)*np.ones(self.Nsrc)

    M1 = simulation.StateMonitor(self.rule, 'out')

    sim1 = simulation.Simulation(self.source, self.motor, self.rule, M1, dt=dt)
    sim1.run(tmax)

    # make sure we normally go outside the range
    self.assertGreater(np.sum(M1.out < mrate), 0)
    self.assertGreater(np.sum(M1.out > Mrate), 0)

    self.rule.compress_rates = True

    M2 = simulation.StateMonitor(self.rule, 'out')

    sim2 = simulation.Simulation(self.source, self.motor, self.rule, M2, dt=dt)
    sim2.run(tmax)

    self.assertEqual(np.sum(M2.out < mrate), 0)
    self.assertEqual(np.sum(M2.out > Mrate), 0)

  def test_compression_tanh(self):
    """ Test that compression performs tanh on uncompressed results. """
    tau = 48.0
    mrate = 60.0
    Mrate = 100.0
    gain = 5

    tmax = 50.0
    dt = 0.2

    self.rule.tau = tau
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False
    self.rule.gain = gain

    self.motor.error_fct = lambda t: (int_r(t/20.0)%3-1)*np.ones(self.Nsrc)

    M1 = simulation.StateMonitor(self.rule, 'out')

    sim1 = simulation.Simulation(self.source, self.motor, self.rule, M1, dt=dt)
    sim1.run(tmax)

    self.rule.compress_rates = True

    M2 = simulation.StateMonitor(self.rule, 'out')

    sim2 = simulation.Simulation(self.source, self.motor, self.rule, M2, dt=dt)
    sim2.run(tmax)

    mavg = 0.5*(mrate + Mrate)
    mdiff = 0.5*(Mrate - mrate)

    expected = mavg + mdiff*np.tanh((M1.out - mavg)/mdiff)

    self.assertTrue(np.allclose(M2.out, expected), msg=
      "mean(abs(out - expected))={}".format(np.mean(np.abs(M2.out - expected))))

  def test_deconvolve_to_motor_error(self):
    """ Test that deconvolving can undo the effect of the memory integral. """
    tau = 50.0
    mrate = 50.0
    Mrate = 100.0

    tmax = 50.0
    dt = 0.1

    self.rule.tau = tau
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False
    self.rule.gain = 1
    self.rule.tau_deconv1 = tau

    self.motor.error_fct = lambda _: np.ones(self.Nsrc)

    M = simulation.StateMonitor(self.rule, 'out')

    sim = simulation.Simulation(self.source, self.motor, self.rule, M, dt=dt)
    sim.run(tmax)
    
    # the output should be almost constant
    self.assertAlmostEqual(np.std(M.out)/np.mean(M.out), 0)

  def test_deconvolve_once_general(self):
    """ Test more general deconvolution timescale. """
    tau = 50.0
    tau_deconv = 20.0
    mrate = 50.0
    Mrate = 100.0

    tmax = 60.0
    dt = 0.1

    self.rule.tau = tau
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False

    self.motor.error_fct = lambda t: (int_r(t/20.0)%3-1)*np.ones(self.Nsrc)

    M1 = simulation.StateMonitor(self.rule, 'out')

    sim1 = simulation.Simulation(self.source, self.motor, self.rule, M1, dt=dt)
    sim1.run(tmax)

    self.rule.tau_deconv1 = tau_deconv

    M2 = simulation.StateMonitor(self.rule, 'out')

    sim2 = simulation.Simulation(self.source, self.motor, self.rule, M2, dt=dt)
    sim2.run(tmax)

    mavg = (mrate + Mrate)*0.5
    mdiff = (Mrate - mrate)*0.5

    out1 = (M1.out - mavg)/mdiff
    out2 = (M2.out - mavg)/mdiff

    der_out1 = np.diff(out1, axis=1)/dt

    expected_out2_crop = out1[:, 1:] + tau_deconv*der_out1

    # mismatch is relatively large since we're using Euler's method
    # we can't do much better, however, since the motor controller cannot give
    # us motor error information at sub-step resolution
    mismatch = np.mean(np.abs(expected_out2_crop - out2[:, 1:])/
        expected_out2_crop)
    self.assertLess(mismatch, 1e-3)

  def test_deconvolve_once_symmetric(self):
    """ Test that it doesn't matter which tau_deconv is non-zero. """
    tau = 50.0
    tau_deconv = 20.0
    mrate = 50.0
    Mrate = 100.0

    tmax = 60.0
    dt = 0.1

    self.rule.tau = tau
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False
    self.rule.tau_deconv1 = tau_deconv
    self.rule.tau_deconv2 = None

    self.motor.error_fct = lambda t: (int_r(t/20.0)%3-1)*np.ones(self.Nsrc)

    M1 = simulation.StateMonitor(self.rule, 'out')

    sim1 = simulation.Simulation(self.source, self.motor, self.rule, M1, dt=dt)
    sim1.run(tmax)

    self.rule.tau_deconv1 = None
    self.rule.tau_deconv2 = tau_deconv

    M2 = simulation.StateMonitor(self.rule, 'out')

    sim2 = simulation.Simulation(self.source, self.motor, self.rule, M2, dt=dt)
    sim2.run(tmax)

    self.assertTrue(np.allclose(M1.out, M2.out))

  def test_deconvolve_second(self):
    """ Test deconvolution with two timescales. """
    tau = 50.0
    tau_deconv1 = 20.0
    tau_deconv2 = 35.0
    mrate = 50.0
    Mrate = 100.0

    tmax = 100.0
    dt = 0.1

    self.rule.tau = tau
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False
    self.rule.tau_deconv1 = tau_deconv1
    self.rule.tau_deconv2 = None

    self.motor.error_fct = lambda t: 2*np.sin(0.123 + t/15.0)*np.ones(self.Nsrc)

    M1 = simulation.StateMonitor(self.rule, 'out')

    sim1 = simulation.Simulation(self.source, self.motor, self.rule, M1, dt=dt)
    sim1.run(tmax)

    self.rule.tau_deconv2 = tau_deconv2

    M2 = simulation.StateMonitor(self.rule, 'out')

    sim2 = simulation.Simulation(self.source, self.motor, self.rule, M2, dt=dt)
    sim2.run(tmax)

    mavg = (mrate + Mrate)*0.5
    mdiff = (Mrate - mrate)*0.5

    out1 = (M1.out - mavg)/mdiff
    out2 = (M2.out - mavg)/mdiff

    der_out1 = np.diff(out1, axis=1)/dt

    expected_out2_crop = out1[:, 1:] + tau_deconv2*der_out1

    # mismatch is relatively large since we're using Euler's method
    # we can't do much better, however, since the motor controller cannot give
    # us motor error information at sub-step resolution
    mismatch = np.mean(np.abs(expected_out2_crop - out2[:, 1:])/
        expected_out2_crop)
    self.assertLess(mismatch, 1e-3)

  def test_deconvolve_symmetric(self):
    """ Test that deconvolution is symmetric in the two timescales. """
    tau = 50.0
    tau_deconv1 = 5.0
    tau_deconv2 = 20.0
    mrate = 50.0
    Mrate = 100.0

    tmax = 60.0
    dt = 0.1

    self.rule.tau = tau
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False
    self.rule.tau_deconv1 = tau_deconv1
    self.rule.tau_deconv2 = tau_deconv2

    self.motor.error_fct = lambda t: 2*np.sin(0.123 + t/15.0)*np.ones(self.Nsrc)

    M1 = simulation.StateMonitor(self.rule, 'out')

    sim1 = simulation.Simulation(self.source, self.motor, self.rule, M1, dt=dt)
    sim1.run(tmax)

    self.rule.tau_deconv1 = tau_deconv2
    self.rule.tau_deconv2 = tau_deconv1

    M2 = simulation.StateMonitor(self.rule, 'out')

    sim2 = simulation.Simulation(self.source, self.motor, self.rule, M2, dt=dt)
    sim2.run(tmax)

    self.assertTrue(np.allclose(M1.out, M2.out))
  
  def test_relaxation_end(self):
    """ Test that relaxation is done after time `relaxation/2`. """
    tau = 50.0
    mrate = 40.0
    Mrate = 120.0

    tmax = 50.0
    dt = 0.1
    relaxation = 20.0

    self.rule.tau = tau
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False
    self.rule.relaxation = relaxation

    self.motor.error_fct = lambda _: np.ones(self.Nsrc)

    M = simulation.StateMonitor(self.rule, 'out')

    sim = simulation.Simulation(self.source, self.motor, self.rule, M, dt=dt)
    sim.run(tmax)

    mask = (M.t > tmax - relaxation/2)
    mavg = 0.5*(mrate + Mrate)

    self.assertAlmostEqual(np.mean(np.abs(M.out[:, mask] - mavg)), 0.0)

  def test_relaxation_no_change_beginning(self):
    """ Test that non-zero `relaxation` doesn't change beginning of run. """
    tau = 50.0
    mrate = 40.0
    Mrate = 120.0

    tmax = 50.0
    dt = 0.1
    relaxation = 20.0

    self.rule.tau = tau
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False

    self.motor.error_fct = lambda _: np.ones(self.Nsrc)

    M1 = simulation.StateMonitor(self.rule, 'out')

    sim1 = simulation.Simulation(self.source, self.motor, self.rule, M1, dt=dt)
    sim1.run(tmax)

    self.rule.relaxation = relaxation
    M2 = simulation.StateMonitor(self.rule, 'out')

    sim2 = simulation.Simulation(self.source, self.motor, self.rule, M2, dt=dt)
    sim2.run(tmax)

    mask = (M1.t < tmax - relaxation)
    self.assertAlmostEqual(np.mean(np.abs(M1.out[:, mask] - M2.out[:, mask])),
        0.0)

    self.assertNotAlmostEqual(np.mean(np.abs(M1.out[:, ~mask] -
        M2.out[:, ~mask])), 0.0)

  def test_relaxation_smooth_monotonic(self):
    """ Test that relaxation is smooth, monotonic, and non-constant. """
    tau = 50.0
    mrate = 40.0
    Mrate = 120.0

    tmax = 100.0
    dt = 0.1
    relaxation = 50.0

    self.rule.tau = tau
    self.rule.min_rate = mrate
    self.rule.max_rate = Mrate
    self.rule.compress_rates = False

    self.motor.error_fct = lambda _: np.ones(self.Nsrc)

    M1 = simulation.StateMonitor(self.rule, 'out')

    sim1 = simulation.Simulation(self.source, self.motor, self.rule, M1, dt=dt)
    sim1.run(tmax)

    self.rule.relaxation = relaxation
    M2 = simulation.StateMonitor(self.rule, 'out')

    sim2 = simulation.Simulation(self.source, self.motor, self.rule, M2, dt=dt)
    sim2.run(tmax)

    mask = ((M1.t >= tmax - relaxation) & (M1.t <= tmax - relaxation/2))

    self.assertAlmostEqual(np.max(np.std(M1.out[:, mask], axis=0)), 0.0)
    self.assertAlmostEqual(np.max(np.std(M2.out[:, mask], axis=0)), 0.0)

    inp = np.mean(M1.out[:, mask], axis=0)
    out = np.mean(M2.out[:, mask], axis=0)

    mavg = 0.5*(mrate + Mrate)
    step_profile = (out - mavg) / (inp - mavg)

    # make sure step is monotonically decreasing, and between 0 and 1
    self.assertTrue(np.all(np.diff(step_profile) < 0))
    self.assertLessEqual(np.max(step_profile), 1.0)
    self.assertGreaterEqual(np.min(step_profile), 0.0)

    # make sure the slope isn't *too* big
    max_diff = float(dt)/(relaxation / 5.0)
    self.assertLess(np.max(-np.diff(step_profile)), max_diff)

##############################
# TestReinforcementTutorRule #
##############################

# helper class
class MockReward(object):
  def __init__(self, reward_fct):
    self.reward_fct = reward_fct

  def prepare(self, t, dt):
    self.reward = 0

  def evolve(self, t, dt):
    self.reward = self.reward_fct(t)

  def __call__(self):
    return self.reward

# actual tests
class TestReinforcementTutorRule(unittest.TestCase):
  def test_direction_no_int(self):
    """ Test that rates change in the expected direction (tau = 0). """
    tmax = 40.0
    dt = 0.1

    tutor = SimpleNeurons(2, out_fct=lambda _: [100.0, 60.0])
    reward = MockReward(lambda t: 1.0 if t < tmax/2 else -1)
    tutor_rule = ReinforcementTutorRule(tutor, reward, tau=0,
        constrain_rates=False, ini_rate=80.0, learning_rate=0.1,
        use_tutor_baseline=False)

    sim = simulation.Simulation(tutor, reward, tutor_rule, dt=dt)
    sim.run(tmax)

    # tutor_rule's output should be increasing for t < tmax/2, decreasing after
    mask = (np.arange(0, tmax, dt) < tmax/2)

    self.assertGreater(np.min(tutor_rule.rates[mask, 0]), 80.0)
    self.assertLess(np.max(tutor_rule.rates[mask, 1]), 80.0)

    self.assertGreater(np.min(tutor_rule.rates[~mask, 1]), 80.0)
    self.assertLess(np.max(tutor_rule.rates[~mask, 0]), 80.0)

  def test_out_follows_rates(self):
    """ Test that rule output follows rates field. """
    tmax = 40.0
    dt = 0.1

    tutor = SimpleNeurons(2, out_fct=lambda _: [100.0, 60.0])
    reward = MockReward(lambda _: 0.0)
    tutor_rule = ReinforcementTutorRule(tutor, reward, tau=0,
        constrain_rates=False, ini_rate=80.0, learning_rate=0.1,
        use_tutor_baseline=False)

    nsteps = int_r(tmax/dt)
    tutor_rule.rates = np.zeros((nsteps, 2))

    tutor_rule.rates[:, 0] = np.linspace(0, 1, nsteps)
    tutor_rule.rates[:, 1] = np.linspace(1, 0, nsteps)

    M = simulation.StateMonitor(tutor_rule, 'out')

    sim = simulation.Simulation(tutor, reward, tutor_rule, M, dt=dt)
    sim.run(tmax)

    self.assertLess(np.max(np.abs(M.out[0] - np.linspace(0, 1, nsteps))), 1e-6)
    self.assertLess(np.max(np.abs(M.out[1] - np.linspace(1, 0, nsteps))), 1e-6)

  def test_prop_learning_rate(self):
    """ Test that rates changes are proportional to learning rate. """
    tmax = 10.0
    dt = 1.0

    learning_rate1 = 0.1
    learning_rate2 = 0.5

    ini_rate = 80.0

    tutor = SimpleNeurons(1, out_fct=lambda _: ini_rate+20.0)
    reward = MockReward(lambda t: 1.0 if t < tmax/2 else -1)
    tutor_rule = ReinforcementTutorRule(tutor, reward, tau=0,
        constrain_rates=False, ini_rate=ini_rate, learning_rate=learning_rate1,
        use_tutor_baseline=False)

    sim1 = simulation.Simulation(tutor, reward, tutor_rule, dt=dt)
    sim1.run(tmax)

    drates1 = tutor_rule.rates - ini_rate

    tutor_rule.reset_rates()
    tutor_rule.learning_rate = learning_rate2

    sim2 = simulation.Simulation(tutor, reward, tutor_rule, dt=dt)
    sim2.run(tmax)

    drates2 = tutor_rule.rates - ini_rate

    self.assertLess(np.max(np.abs(learning_rate2*drates1 -
        learning_rate1*drates2)), 1e-6)

  def test_prop_reward(self):
    """ Test that rates changes scale linearly with reward. """
    tmax = 10.0
    dt = 1.0

    reward_scale = 5.0

    ini_rate = 80.0

    tutor = SimpleNeurons(1, out_fct=lambda _: ini_rate+20.0)
    reward = MockReward(lambda t: 1.0 if t < tmax/2 else -1)
    tutor_rule = ReinforcementTutorRule(tutor, reward, tau=0,
        constrain_rates=False, ini_rate=ini_rate, learning_rate=1.0,
        use_tutor_baseline=False)

    sim1 = simulation.Simulation(tutor, reward, tutor_rule, dt=dt)
    sim1.run(tmax)

    drates1 = tutor_rule.rates - ini_rate

    tutor_rule.reset_rates()
    reward.reward_fct =  lambda t: reward_scale if t < tmax/2 else -reward_scale

    sim2 = simulation.Simulation(tutor, reward, tutor_rule, dt=dt)
    sim2.run(tmax)

    drates2 = tutor_rule.rates - ini_rate

    self.assertLess(np.max(np.abs(reward_scale*drates1 - drates2)), 1e-6)

  def test_prop_fluctuation(self):
    """ Test that rates changes scale linearly with fluctuation size. """
    tmax = 10.0
    dt = 1.0

    ini_rate = 80.0

    nsteps = int_r(tmax/dt)

    tutor = SimpleNeurons(1, out_fct=lambda i: ini_rate + i*20.0/nsteps - 10.0)
    reward = MockReward(lambda _: 1.0)
    tutor_rule = ReinforcementTutorRule(tutor, reward, tau=0,
        constrain_rates=False, ini_rate=ini_rate, learning_rate=1.0,
        use_tutor_baseline=False)

    sim = simulation.Simulation(tutor, reward, tutor_rule, dt=dt)
    sim.run(tmax)

    drates = (tutor_rule.rates - ini_rate)[:, 0]

    fluctuations = (np.arange(nsteps)*20.0/nsteps - 10.0)
    mask = (fluctuations > 0)
    ratio = np.mean(drates[mask] / fluctuations[mask])

    self.assertLess(np.max(np.abs(drates - ratio*fluctuations)), 1e-6)

  def test_constrain_rates(self):
    """ Test that we can keep rates constrained in a given range. """
    tmax = 10.0
    dt = 1.0

    ini_rate = 80.0
    min_rate = ini_rate - 5.0
    max_rate = ini_rate + 5.0

    nsteps = int_r(tmax/dt)

    tutor = SimpleNeurons(1, out_fct=lambda i: ini_rate + i*20.0/nsteps - 10.0)
    reward = MockReward(lambda _: 1.0)
    tutor_rule = ReinforcementTutorRule(tutor, reward, tau=0,
        constrain_rates=False, ini_rate=ini_rate, learning_rate=1.0,
        min_rate=min_rate, max_rate=max_rate,
        use_tutor_baseline=False)

    sim1 = simulation.Simulation(tutor, reward, tutor_rule, dt=dt)
    sim1.run(tmax)

    # rates should exceed limits
    self.assertGreater(np.max(tutor_rule.rates), max_rate)
    self.assertLess(np.min(tutor_rule.rates), min_rate)

    tutor_rule.constrain_rates = True
    tutor_rule.reset_rates()

    sim2 = simulation.Simulation(tutor, reward, tutor_rule, dt=dt)
    sim2.run(tmax)
    
    # rates should no longer exceed limits
    self.assertLessEqual(np.max(tutor_rule.rates), max_rate)
    self.assertGreaterEqual(np.min(tutor_rule.rates), min_rate)

  def test_tau(self):
    """ Test integrating the reward-fluctuation product over some timescale. """
    tau_values = [5.0, 15.0, 25.0]
    
    tmax = 50.0
    dt = 0.1
    N = 3

    ini_rate = 80.0

    nsteps = int_r(tmax/dt)

    # reproducible arbitrariness
    np.random.seed(34342)

    tutor_out_trace = ini_rate + 20.0*np.random.randn(nsteps, N)
    # have some correlation between reward trace and tutor.out trace
    rho = 0.2
    reward_trace = (rho*(tutor_out_trace[:, 0] - ini_rate)/20.0 +
        (1-rho)*np.random.randn(nsteps))
    
    scaling = None

    for crt_tau in tau_values:
      tutor = SimpleNeurons(N, out_fct=lambda i: tutor_out_trace[i])
      reward = MockReward(lambda t: reward_trace[int_r(t/dt)])
      tutor_rule = ReinforcementTutorRule(tutor, reward, tau=crt_tau,
          constrain_rates=False, ini_rate=ini_rate, learning_rate=1.0,
          use_tutor_baseline=False)

      sim = simulation.Simulation(tutor, reward, tutor_rule, dt=dt)
      sim.run(tmax)

      drates = tutor_rule.rates - ini_rate

      # this should be a convolution of tutor_out_trace*reward_trace with an
      # exponential with time constant crt_tau
      # that means that tau*(d/dt)drates + drates must be proportional to it
      expected_rhs = (tutor_out_trace - ini_rate)*np.reshape(reward_trace,
          (-1, 1))

      lhs = np.vstack((float(crt_tau)*np.reshape(drates[0, :], (1, -1))/dt,
          (crt_tau/dt)*np.diff(drates, axis=0) + drates[:-1, :]))
      
      # allow scaling to be arbitrary, but *independent of tau*
      if scaling is None:
        mask = (expected_rhs != 0)
        scaling = np.mean(lhs[mask]/expected_rhs[mask])

        # scaling shouldn't be negative or zero!
        self.assertGreater(scaling, 1e-9)

      mag = np.mean(np.abs(expected_rhs))

      self.assertLess(np.max(np.abs(lhs - scaling*expected_rhs)), 1e-6*mag)

  def test_use_tutor_baseline(self):
    """ Test using average tut. rates instead of intended ones as reference. """
    tmax = 40.0
    dt = 1.0
    ini_rate = 80.0

    nruns = 11

    tutor = SimpleNeurons(2, out_fct=lambda _: [100.0, 60.0])
    reward = MockReward(lambda t: 1.0 if t < tmax/2 else -1)
    tutor_rule = ReinforcementTutorRule(tutor, reward, tau=0,
        constrain_rates=False, ini_rate=ini_rate, learning_rate=0.1)

    tutor_rule.use_tutor_baseline = True
    tutor_rule.baseline_n = 5

    for i in xrange(nruns):
      # we first set the baselines for the two neurons to some values different
      # from tutor_rule's ini_rate, and then in the last round, we test how the
      # rates change
      if i == nruns-1:
        tutor.out_fct = lambda _: [80.0, 80.0]

      tutor_rule.reset_rates()

      sim = simulation.Simulation(tutor, reward, tutor_rule, dt=dt)
      sim.run(tmax)

    drates = tutor_rule.rates - ini_rate

    # for the first neuron, for t < tmax/2, the current firing rate is below the
    # baseline and the reward is positive, so the rates should *decrease*
    # for t >= tmax/2, the rates should *increase*
    # the opposite should happen for the second neuron
    mask = (np.arange(0, tmax, dt) < tmax/2)

    self.assertGreater(np.min(drates[mask, 1]), 0)
    self.assertLess(np.max(drates[mask, 0]), 0)

    self.assertGreater(np.min(drates[~mask, 0]), 0)
    self.assertLess(np.max(drates[~mask, 1]), 0)

  def test_calculate_tutor_baseline(self):
    """ Test calculation of average tutor rates. """
    tmax = 40.0
    dt = 1.0
    ini_rate = 80.0
    baseline_n = 5

    rate1 = ini_rate + 20.0
    rate2 = ini_rate - 10.0

    nruns = 10
    nsteps = int_r(tmax/dt)

    tutor = SimpleNeurons(2, out_fct=lambda i:
        [rate1, rate2] if i < nsteps/2 else [rate2, rate1])
    reward = MockReward(lambda _: 0.0)
    tutor_rule = ReinforcementTutorRule(tutor, reward, tau=0,
        constrain_rates=False, ini_rate=ini_rate, learning_rate=0.1,
        use_tutor_baseline=True, baseline_n=baseline_n)

    factor = 1 - 1.0/baseline_n

    for i in xrange(nruns):
      tutor_rule.reset_rates()

      sim = simulation.Simulation(tutor, reward, tutor_rule, dt=dt)
      sim.run(tmax)

      crt_baseline = tutor_rule.baseline

      self.assertEqual(np.ndim(crt_baseline), 2)
      self.assertEqual(np.shape(crt_baseline)[0], nsteps)
      self.assertEqual(np.shape(crt_baseline)[1], 2)

      expected1 = rate1 + (ini_rate - rate1)*factor**(i+1)
      expected2 = rate2 + (ini_rate - rate2)*factor**(i+1)

      self.assertLess(np.max(np.abs(crt_baseline[:nsteps/2, 0] - expected1)),
          1e-6)
      self.assertLess(np.max(np.abs(crt_baseline[nsteps/2:, 0] - expected2)),
          1e-6)

      self.assertLess(np.max(np.abs(crt_baseline[:nsteps/2, 1] - expected2)),
          1e-6)
      self.assertLess(np.max(np.abs(crt_baseline[nsteps/2:, 1] - expected1)),
          1e-6)

  def test_relaxation_end(self):
    """ Test that relaxation is done after time `relaxation/2`. """
    tau = 50.0
    mrate = 40.0
    Mrate = 120.0

    tmax = 50.0
    dt = 0.1
    relaxation = 20.0

    tutor = SimpleNeurons(2, out_fct=lambda _: Mrate*np.random.rand())
    reward = MockReward(lambda t: np.sin(10*t/tmax))
    tutor_rule = ReinforcementTutorRule(tutor, reward, tau=tau,
        constrain_rates=True, min_rate=mrate, max_rate=Mrate,
        learning_rate=0.1, relaxation=relaxation, use_tutor_baseline=False)

    # reproducible arbitrariness
    np.random.seed(1)

    M = simulation.StateMonitor(tutor_rule, 'out')

    sim = simulation.Simulation(tutor, reward, tutor_rule, M, dt=dt)
    sim.run(tmax)

    mask = (M.t > tmax - relaxation/2)
    mavg = 0.5*(mrate + Mrate)

    self.assertAlmostEqual(np.mean(np.abs(M.out[:, mask] - mavg)), 0.0)

  def test_relaxation_no_change_beginning(self):
    """ Test that non-zero `relaxation` doesn't change beginning of run. """
    tau = 25.0
    mrate = 40.0
    Mrate = 120.0

    tmax = 50.0
    dt = 0.1
    relaxation = 20.0

    tutor = SimpleNeurons(2, out_fct=lambda _: Mrate*np.random.rand())
    reward = MockReward(lambda t: np.sin(8*t/tmax))
    tutor_rule = ReinforcementTutorRule(tutor, reward, tau=tau,
        constrain_rates=True, min_rate=mrate, max_rate=Mrate,
        learning_rate=0.1, relaxation=None, use_tutor_baseline=False)

    # reproducible arbitrariness
    np.random.seed(12)

    M1 = simulation.StateMonitor(tutor_rule, 'out')

    sim1 = simulation.Simulation(tutor, reward, tutor_rule, M1, dt=dt)
    sim1.run(tmax)

    # now run again with relaxation enabled
    tutor_rule.relaxation = relaxation
    tutor_rule.reset_rates()
    np.random.seed(12)

    M2 = simulation.StateMonitor(tutor_rule, 'out')

    sim2 = simulation.Simulation(tutor, reward, tutor_rule, M2, dt=dt)
    sim2.run(tmax)

    mask = (M1.t < tmax - relaxation)
    self.assertAlmostEqual(np.mean(np.abs(M1.out[:, mask] - M2.out[:, mask])),
        0.0)

    self.assertNotAlmostEqual(np.mean(np.abs(M1.out[:, ~mask] -
        M2.out[:, ~mask])), 0.0)

  def test_relaxation_smooth_monotonic(self):
    """ Test that relaxation is smooth, monotonic, and non-constant. """
    tau = 45.0
    mrate = 40.0
    Mrate = 120.0

    tmax = 50.0
    dt = 0.1
    relaxation = 20.0

    tutor = SimpleNeurons(2, out_fct=lambda _: Mrate*np.random.rand())
    reward = MockReward(lambda t: np.sin(9*t/tmax))
    tutor_rule = ReinforcementTutorRule(tutor, reward, tau=tau,
        constrain_rates=True, min_rate=mrate, max_rate=Mrate,
        learning_rate=0.1, relaxation=None, use_tutor_baseline=False)

    # reproducible arbitrariness
    np.random.seed(123)

    M1 = simulation.StateMonitor(tutor_rule, 'out')

    sim1 = simulation.Simulation(tutor, reward, tutor_rule, M1, dt=dt)
    sim1.run(tmax)

    # now run again with relaxation enabled
    tutor_rule.relaxation = relaxation
    tutor_rule.reset_rates()
    np.random.seed(123)

    M2 = simulation.StateMonitor(tutor_rule, 'out')

    sim2 = simulation.Simulation(tutor, reward, tutor_rule, M2, dt=dt)
    sim2.run(tmax)

    mask = ((M1.t >= tmax - relaxation) & (M1.t <= tmax - relaxation/2))

    mavg = 0.5*(mrate + Mrate)
    ratio = (M2.out[:, mask] - mavg)/(M1.out[:, mask] - mavg)
    self.assertAlmostEqual(np.max(np.std(ratio, axis=0)), 0.0)

    step_profile = np.mean(ratio, axis=0)

    # make sure step is monotonically decreasing, and between 0 and 1
    self.assertTrue(np.all(np.diff(step_profile) < 0))
    self.assertLessEqual(np.max(step_profile), 1.0)
    self.assertGreaterEqual(np.min(step_profile), 0.0)

    # make sure the slope isn't *too* big
    max_diff = float(dt)/(relaxation / 5.0)
    self.assertLess(np.max(-np.diff(step_profile)), max_diff)

##################
# TestRateLayer  #
##################

class TestRateLayer(unittest.TestCase):

  def test_linear(self):
    """ Test layer in linear regime. """
    G1 = SimpleNeurons(3)
    G2 = SimpleNeurons(2)

    G1pattern = np.asarray(
        [[ 0, 1, 0, 2],
         [-1, 1, 0, 1],
         [ 1,-1,-1,-1]])
    G2pattern = np.asarray(
        [[0,      1,      4,      0],
         [1, -1.0/3, -1.0/3, -2.0/3]])

    G1.out_fct = lambda i: G1pattern[:, i]
    G2.out_fct = lambda i: G2pattern[:, i]

    G = RateLayer(2)
    G.add_source(G1)
    G.add_source(G2)

    G.Ws[0] = np.asarray(
        [[1, 2, 3],
         [1,-2, 1]])
    G.Ws[1] = np.asarray([1, -3])

    M = simulation.StateMonitor(G, 'out')

    dt = 1.0
    nsteps = 4
    tmax = nsteps*dt

    sim = simulation.Simulation(G1, G2, G, M, dt=dt)
    sim.run(tmax)

    self.assertTrue(np.allclose(M.out[0, :], [1, 1, 1, 1]))
    self.assertTrue(np.allclose(M.out[1, :], [0, -1, 0, 1]))

  def test_nonlinearity(self):
    """ Test application of nonlinearity. """
    # reproducible arbitrariness
    np.random.seed(1)

    N1 = 5
    N2 = 4
    N = 3

    dt = 1.0
    nsteps = 10
    tmax = nsteps*dt

    nonlin = lambda v: np.tanh(v)

    G1 = SimpleNeurons(N1)
    G2 = SimpleNeurons(N2)

    G1pattern = 1 + 2*np.random.randn(N1, nsteps)
    G2pattern = -1 + np.random.randn(N2, nsteps)
    G1.out_fct = lambda i: G1pattern[:, i]
    G2.out_fct = lambda i: G2pattern[:, i]

    G = RateLayer(N)

    G.add_source(G1)
    G.add_source(G2)

    G.Ws[0] = np.random.randn(N, N1)
    G.Ws[1] = 1 + 3*np.random.randn(N, N2)

    M1 = simulation.StateMonitor(G, 'out')
    sim1 = simulation.Simulation(G1, G2, G, M1, dt=dt)
    sim1.run(tmax)

    # test that the run isn't trivial
    self.assertGreater(np.mean(np.abs(M1.out)), 1e-3)

    # now run again with nonlinearity
    G.nonlinearity = nonlin

    M2 = simulation.StateMonitor(G, 'out')
    sim2 = simulation.Simulation(G1, G2, G, M2, dt=dt)
    sim2.run(tmax)

    self.assertTrue(np.allclose(M2.out, nonlin(M1.out)))

  def test_bias(self):
    """ Test neuron bias. """
    N = 4

    G = RateLayer(N)

    bias = [1.0, 0.0, -1.0, -2.0]
    G.bias = np.array(bias)

    M = simulation.StateMonitor(G, 'out')
    sim = simulation.Simulation(G, M, dt=1.0)
    sim.run(sim.dt)

    self.assertTrue(np.allclose(M.out.ravel(), bias))

####################
# TestRateHVCLayer #
####################

class TestRateHVCLayer(unittest.TestCase):
  def test_jitter(self):
    """ Test that there are differences in output between trials. """
    # some reproducible arbitrariness
    np.random.seed(343143)

    n = 25
    t_max = 50
    dt = 0.1
    G = RateHVCLayer(n)

    M1 = simulation.StateMonitor(G, 'out')

    sim1 = simulation.Simulation(G, M1, dt=dt)
    sim1.run(t_max)
    
    M2 = simulation.StateMonitor(G, 'out')
    sim2 = simulation.Simulation(G, M2, dt=dt)
    sim2.run(t_max)

    self.assertGreater(np.max(np.abs(M1.out - M2.out)), 0.99)

  def test_no_jitter(self):
    """ Test that repeated noiseless trials are identical. """
    n = 10
    t_max = 25
    dt = 0.1
    G = RateHVCLayer(n)
    G.burst_noise = 0.0

    M1 = simulation.StateMonitor(G, 'out')

    sim1 = simulation.Simulation(G, M1, dt=dt)
    sim1.run(t_max)
    
    M2 = simulation.StateMonitor(G, 'out')
    sim2 = simulation.Simulation(G, M2, dt=dt)
    sim2.run(t_max)

    self.assertTrue(np.allclose(M1.out, M2.out))

  def test_uniform(self):
    """ Test that there are bursts all along the simulation window. """
    # some reproducible arbitrariness
    np.random.seed(87548)

    n = 50
    t_max = 50
    dt = 0.1
    resolution = 1.0

    class UniformityChecker(object):
      def __init__(self, target, resolution):
        self.target = target
        self.resolution = resolution
        self.order = 1

      def prepare(self, t_max, dt):
        self.has_spike = np.zeros(int_r(t_max/self.resolution) + 1)

      def evolve(self, t, dt):
        i = int_r(t/self.resolution)
        self.has_spike[i] = (self.has_spike[i] or np.any(self.target.out > 0))

    G = RateHVCLayer(n)
    M = UniformityChecker(G, resolution)
    sim = simulation.Simulation(G, M, dt=dt)
    sim.run(t_max)

    self.assertTrue(np.all(M.has_spike))

  def test_burst(self):
    """ Test that each neuron fires a burst of given width. """
    n = 25
    t_max = 50
    dt = 0.1

    G = RateHVCLayer(n)
    G.burst_noise = 0.0

    M = simulation.StateMonitor(G, 'out')
    sim = simulation.Simulation(G, M, dt=dt)
    sim.run(t_max)

    # find burst lengths for each neuron
    nonzero_len = lambda v: max((v>0).nonzero()[0]) - min((v>0).nonzero()[0])
    burst_lengths = [dt*nonzero_len(M.out[i]) for i in xrange(n)]

    self.assertLess(np.std(burst_lengths), dt/2)
    self.assertLess(np.abs(np.mean(burst_lengths) - float(t_max)/n), 
        (1 + 1e-6)*dt)

  def test_burst_dispersion(self):
    """ Test that starting times of bursts are within required bounds. """
    # some reproducible arbitrariness
    np.random.seed(7342642)

    n = 25
    t_max = 50
    dt = 0.1
    n_sim = 10
    
    G = RateHVCLayer(n)

    burst_starts = []
    for i in xrange(n_sim):
      M = simulation.StateMonitor(G, 'out')
      sim = simulation.Simulation(G, M, dt=dt)
      sim.run(t_max)

      burst_starts.append([dt*min((M.out[i] > 0).nonzero()[0])
        for i in xrange(n)])

    burst_starts_range = [np.ptp([_[i] for _ in burst_starts])
        for i in xrange(n)]
    
    self.assertLess(np.max(burst_starts_range), G.burst_noise + dt/2)

  def test_burst_tmax(self):
    """ Test using a different end time for bursts than for simulation. """
    n = 10
    t_max = 25
    dt = 0.1
    G = RateHVCLayer(n)
    G.burst_noise = 0.0

    M1 = simulation.StateMonitor(G, 'out')

    sim1 = simulation.Simulation(G, M1, dt=dt)
    sim1.run(t_max)

    G = RateHVCLayer(n, burst_tmax=t_max)
    G.burst_noise = 0.0
    
    M2 = simulation.StateMonitor(G, 'out')
    sim2 = simulation.Simulation(G, M2, dt=dt)
    sim2.run(2*t_max)

    self.assertTrue(np.allclose(M1.out, M2.out[:, :M1.out.shape[1]]))

  def test_custom_length(self):
    """ Test custom burst length. """
    n = 25
    t_max = 50
    dt = 0.1
    burst_length = 10.0

    G = RateHVCLayer(n, burst_length=burst_length)
    G.burst_noise = 0.0

    M = simulation.StateMonitor(G, 'out')
    sim = simulation.Simulation(G, M, dt=dt)
    sim.run(t_max)

    # find burst lengths for each neuron
    nonzero_len = lambda v: max((v>0).nonzero()[0]) - min((v>0).nonzero()[0])
    burst_lengths = [dt*nonzero_len(M.out[i]) for i in xrange(n)]

    self.assertLess(np.std(burst_lengths), dt/2)
    self.assertLess(np.abs(np.mean(burst_lengths) - burst_length), 
        (1 + 1e-6)*dt)

####################
# TestTableLayer   #
####################

class TestTableLayer(unittest.TestCase):
  def test_output(self):
    """ Test that the output matches the table provided. """
    # reproducible arbitrariness
    np.random.seed(123423)

    N = 20
    tmax = 30.0
    dt = 1.0

    n_steps = int_r(tmax/dt)

    table = np.random.randn(N, n_steps)
    G = TableLayer(table)

    M = simulation.StateMonitor(G, 'out')

    sim = simulation.Simulation(G, M, dt=dt)
    sim.run(tmax)

    self.assertLess(np.max(np.abs(M.out - table)), 1e-6)

####################
# TestConnector    #
####################

class TestConnector(unittest.TestCase):
  def test_transfer(self):
    """ Test transfer of data using connector. """
    class MockSender(object):
      def evolve(self, t, dt):
        self.x = t**2 + t + 1

    class MockReceiver(object):
      def __init__(self):
        self.rec = None

      def evolve(self, t, dt):
        pass

    S = MockSender()
    R = MockReceiver()
    C = Connector(S, 'x', R, 'rec', order=0.5)

    tmax = 10.0
    dt = 0.1

    M = simulation.StateMonitor(R, 'rec')

    sim = simulation.Simulation(S, R, C, M, dt=dt)
    sim.run(tmax)

    expected = (M.t**2 + M.t + 1)

    self.assertAlmostEqual(np.mean(np.abs(expected - M.rec[0])), 0.0)


if __name__ == '__main__':
  unittest.main()
