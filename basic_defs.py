""" Define classes specifying the conductor-tutor-student simulation. """

import numpy as np

import simulation

def int_r(f):
  """ Convert to nearest integer. """
  return int(np.round(f))

class TableSpikers(object):
  
  """ A layer of neurons that spike at times indicated by a boolean array.
  
  Set `spike_table` (for example by overloading `prepare` method) to decide when
  firing should happen.

  Attributes
  ----------
    spike_table: (n_steps, n_neurons) matrix
      Boolean matrix identifying the steps when each neuron fires.
  """

  def __init__(self, N, tau_out=5.0):
    """ Initialize a layer with `N` neurons. """
    self.N = N
    self.spike_table = None
    self.tau_out = tau_out

  def prepare(self, tmax, dt):
    self.out = np.zeros(self.N)

  def evolve(self, t, dt):
    if self.spike_table is not None:
      i = int_r(t/dt)
      if i < len(self.spike_table):
        self.spike = self.spike_table[i]
      else:
        self.spike = np.zeros(self.N, dtype=bool)
    
    self.out += (1000.0*self.spike - self.out*dt)/self.tau_out

class HVCLikeLayer(TableSpikers):

  """ A layer of neurons each of which fires only once during the program. """

  def __init__(self, N, burst_tmax=None):
    """ Initialize the layer with `N` neurons.
    
    If `burst_tmax` is provided, the neurons will fire up to the time
    `burst_tmax` as opposed to the full length of the simulation.
    """
    super(HVCLikeLayer, self).__init__(N)

    self.spikes_per_burst = 5
    self.rate_during_burst = 650.0 # Hz
    self.burst_noise = 0.6 # ms (amount of noise in burst start times)
    self.spike_noise = 0.4 # ms (amount of noise in within-burst intervals)
    self.burst_tmax = burst_tmax

  def prepare(self, tmax, dt):
    """ Generate the spike pattern for a new run. Distribute the bursts
    uniformly over the time of the simulation.
    """
    super(HVCLikeLayer, self).prepare(tmax, dt)

    burst_tmax = self.burst_tmax
    if burst_tmax is None:
      burst_tmax = tmax

    # time between spikes (ignoring noise)
    spike_dt = 1000.0/self.rate_during_burst

    # total duration of burst (ignoring noise)
    burst_length = spike_dt*(self.spikes_per_burst - 1)

    # figure out when the spikes should be
    burst_start_times = np.linspace(0.0, burst_tmax - burst_length, self.N) + \
        self.burst_noise*(np.random.rand(self.N) - 0.5)
    spike_intervals = spike_dt + self.spike_noise*(np.random.rand(
        self.spikes_per_burst - 1, self.N) - 0.5)
    spike_times = np.clip(burst_start_times + np.cumsum(np.vstack((
        np.zeros(self.N), spike_intervals)), axis=0), 0.0, burst_tmax - dt)

    # figure out the boolean spike table
    nsteps = int_r(burst_tmax/dt)
    self.spike_table = np.zeros((nsteps, self.N), dtype=bool)
    for i in xrange(self.N):
      crt_idxs = np.round(spike_times[:, i]/dt).astype(int)
      self.spike_table[crt_idxs, i] = True

class RandomLayer(TableSpikers):

  """ A layer of neurons that fire Possion spikes with a time-dependent rate.
  """

  def __init__(self, N, ini_rate=80.0):
    """ Initialize the layer with `N` neurons, with an initial firing rate
    given by `ini_rate`. """
    super(RandomLayer, self).__init__(N)

    self.ini_rate = ini_rate
    self.reset_rate()

  def reset_rate(self, new_rate=None):
    """ Reset the firing rates to the initial rate set at construction, or to
    `new_rate`, if provided.
    """
    if new_rate is not None:
      self.ini_rate = new_rate

    self.rate = self.ini_rate

  def prepare(self, tmax, dt):
    super(RandomLayer, self).prepare(tmax, dt)

    nsteps = int_r(tmax/dt)

    if np.size(self.rate) == 1:
      self.rate = self.rate*np.ones((nsteps, self.N))

    if np.ndim(self.rate) == 1:
      self.rate = np.tile(self.rate, ((nsteps, 1)))
    
    self.rate = np.asarray(self.rate)
    self.out = np.copy(self.rate[0])

    self.spike_table = np.random.rand(nsteps, self.N) < self.rate*dt/1000.0

class StudentLayer(object):

  """ A layer of student neurons. """

  def __init__(self, N, vR=-70.0, v_th=-55.0, R=260.0, tau_m=15.0,
               tau_ampa=5.0, tau_nmda=100.0, i_ext=0.0, g_inh=0.0,
               tau_inh=20.0, tau_ref=1.0, tau_out=5.0):
    """ Initialize the layer.

    Arguments
    ---------
      N: int
        Number of neurons in layer.
      vR: float
        Resting potential (mV).
      v_th: float
        Threshold potential (mV).
      R: float
        Input resistance (MOhm).
      tau_m: float
        Membrane time constant (ms).
      tau_ref: float
        Refractory period (ms).
      tau_ampa: float
        AMPA time constant (ms).
      tau_nmda: float
        NMDA time constant (ms).
      i_ext: float
        External current (nA). This can be a function, `fct(student)`, that
        takes the instance of the student layer as argument and returns the
        value of the external current. The function is called from `prepare`.
      g_inh: float
        Strength of global inhibition (mV/Hz).
      tau_inh: float
        Time constant used for inhibition (ms).
      tau_out: float
        Time constant for output (ms). This is not directly used in the
        dynamics.
    """
    self.N = N
    self.vR = vR
    self.v_th = v_th
    self.R = R
    self.tau_m = tau_m
    self.tau_ref = tau_ref
    self.tau_ampa = tau_ampa
    self.tau_nmda = tau_nmda
    self.i_ext_init = i_ext
    self.g_inh = g_inh
    self.tau_inh = tau_inh
    self.tau_out = tau_out

  def prepare(self, tmax, dt):
    if hasattr(self.i_ext_init, '__call__'):
      self.i_ext = self.i_ext_init(self)
    else:
      self.i_ext = self.i_ext_init

    # set up the initial state
    if hasattr(self, 'v_init'):
      self.v = self.v_init
      if hasattr(self.v, '__call__'):
        self.v = self.v(self)
      if np.size(self.v) == 1 and self.N > 1:
        self.v = np.repeat(self.v, self.N)
    else:
      self.v = self.vR*np.ones(self.N)

    self.i_ampa = np.zeros(self.N)
    self.i_nmda = np.zeros(self.N)

    self.dv_inh = 0.0

    self.out = np.zeros(self.N)

    self.spike = np.zeros(self.N, dtype=bool)

    # these are False for neurons in refractory period
    self.active = np.ones(self.N, dtype=bool)
    # these are the times when refractory periods will end
    self.ref_end = np.zeros(self.N)
  
  def evolve(self, t, dt):
    mask = self.active
    # reactivate neurons that are coming out of their refractory periods
    mask[t > self.ref_end] = True

    derivative = (self.vR - self.v + self.R*(self.i_ampa + self.i_nmda +
        self.i_ext) - self.dv_inh)*(1.0/self.tau_m)
    self.v[mask] += derivative[mask]*dt

    # find spikes
    self.spike = (self.v > self.v_th)
    self.v[self.spike] = self.vR

    # update currents
    self.i_ampa -= self.i_ampa*(dt/self.tau_ampa)
    self.i_nmda -= self.i_nmda*(dt/self.tau_nmda)

    # update output
    self.out += (1000.0*self.spike - self.out*dt)/self.tau_out

    # start refractory periods
    self.active[self.spike] = False
    self.ref_end[self.spike] = t + self.tau_ref

    # keep track of global inhibitory signal, if necessary
    if self.g_inh > 0:
      self.dv_inh += (1000.0*self.g_inh/self.N/self.tau_inh)*np.sum(self.spike)\
          - self.dv_inh*(dt/self.tau_inh)

class ExcitatorySynapses(object):

  """ A model for AMPA and NMDA synapses.

  The initial values for `i_ampa` and `i_nmda` in the target layer can be set to
  reasonable values provided the source layer has a member `avg_rates`
  identifying the expected average firing rate(s) for the source neurons. This
  also uses the time constants for AMPA and NMDA integration at the target,
  `target.tau_ampa` and `target.tau_nmda`.
  
  Attributes
  ----------
    source:
    target:
      Source and target layers. The target should have an `i_ampa` current and
      an `active` mask, and also an `i_nmda` current and a  `v` membrane
      potential if there are any NMDA receptors. The source should have a
      `spike` event.
    W: matrix (size `(target.N, source.N)`), or vector
      Connectivity and weight matrix. This can also be a vector, in which case
      the connections are assumed to be one to one. The source and target layer
      would have to have the same number of units in this case.
    f_nmda: float
      Fraction of NMDA receptors.
    mg: float
      Intracellular Mg concentration (in mM).
  """

  def __init__(self, source, target, f_nmda=0.0, one_to_one=False):
    self.source = source
    self.target = target
    self.f_nmda = f_nmda
    self.mg = 0.7
    if one_to_one:
      if self.target.N != self.source.N:
        raise Exception('Source and target layers must have the same ' +
          'size for one-to-one synapses.')
      self.W = np.zeros(self.target.N)
    else:
      self.W = np.zeros((self.target.N, self.source.N))
    self.order = 1

    self.change_during_ref = True

  def prepare(self, t, dt):
    avg_rates = getattr(self.source, 'avg_rates', 0)
    tau_ampa = getattr(self.target, 'tau_ampa', 0)
    tau_nmda = getattr(self.target, 'tau_nmda', 0)
    v_ref = getattr(self.target, 'v_ref', -60.0)

    if np.size(avg_rates) == 1:
      avg_rates = np.repeat(avg_rates, self.source.N)

    if np.ndim(self.W) == 1:
      effect = self.W*avg_rates
    else:
      effect = np.dot(self.W, avg_rates)

    self.target.i_ampa = (tau_ampa*(1 - self.f_nmda)/1000.0)*effect
    if self.f_nmda > 0.0:
      self.target.i_nmda = (tau_nmda*self.f_nmda/1000.0)*effect* \
          self.nmda_factor(v_ref)

  def nmda_factor(self, v):
    return 1.0/(1 + self.mg/3.57*np.exp(-v/16.13))
  
  def evolve(self, t, dt):
    # XXX perhaps I should do the calculation with only the neurons that spiked?
    # XXX and *for* only the ones that were active?
    if np.ndim(self.W) == 1:
      effect = self.W*self.source.spike
    else:
      effect = np.dot(self.W, self.source.spike)

    if self.change_during_ref:
      active_effect = effect
      v = self.target.v
      i_nmda = self.target.i_nmda
      i_ampa = self.target.i_ampa
    else:
      active_mask = self.target.active
      active_effect = effect[active_mask]
      v = self.target.v[active_mask]
      i_nmda = self.target.i_nmda[active_mask]
      i_ampa = self.target.i_ampa[active_mask]

    if self.f_nmda > 0.0:
      # modulation of NMDA current
      G = self.nmda_factor(v)
      i_nmda += self.f_nmda*G*active_effect

    i_ampa += (1.0 - self.f_nmda)*active_effect

class LinearController(object):

  """ A class that manages the mapping between student neurons and output
  channels in a very simple way: by assuming it is linear.

  Attributes
  ----------
    source: neural layer
      The source of activity that is driving the output.
    N: int
      Number of output channels.
    target: sequence of `N` arrays
      The target output(s).
    tau: float
      Smoothing timescale for the output.
    W: array of size `(self.N, self.source.N)`
      The linear map relating source activities to outputs.
    bias: array of size `self.N`
      Output biases. Output is given schematically by
        y = b + W*(smoothed source)
    permute_inverse: None or array
      If not `None`, specifies a permutation of the source neuron that modifies
      only the output of `get_source_error`, but not the actual output of the
      `LinearController`.
    error_map_fct: None, or function
      If not `None`, specifies a function (nonlinearity) that is applied to the
      motor error before being multiplied by the weight matrix to get the source
      error.
    nonlinearity: None, or function
      if not `None`, specifies a function (nonlinearity) that is applied to the
      weighted input before setting the output, i.e.,
        u = nonlinearity(b + W*(smoothed source))
      This nonlinearity is not taken into account when calculating motor errors.
  """

  def __init__(self, source, target, mode='sum', factor=1.0, tau=30.0):
    """ Initialize the controller, with some default mapping.

    Arguments
    ---------
      source: neural layer
        Source layer. Should contain an `out` member.
      target: int or array of arrays
        If `int`, number of output channels. Otherwise this is the target motor
        program. If only the number of channels is given, target is set to all
        zeros.
      mode: string
        Kind of initialization to use for the mapping. The source neurons are
        split into equal subsets, and then weights are assigned as follows: if
        `mode` is
          'sum':      all the weights are equal, so that the activity of the
                      source neurons corresponding to each channel is simply
                      additive.
          'pushpull': each subgroup is further split into two parts, one with
                      positive weights and one with negative weights, so that
                      the source neurons act in a push-pull manner.
          'zero':     leave the weights zero for now.
        Behavior is undefined if the number of source neurons is not an integer
        multiple of the number of channels, `N`. In all cases, the weights can
        later be altered by changing the `W` member.
      factor: float
        Factor by which to multiply the weights. By default all the weights
        scale inversely-proportional to `source.N`.
      tau: float
        Timescale for smoothing of output. Set to 0 or None to perform no
        smoothing.
    """
    if np.size(target) == 1:
      self.N = target
      self.target = np.zeros((self.N, 1))
    else:
      self.N = len(target)
      self.target = np.asarray(target)

    self.source = source
    self.tau = tau   # ms
    self.bias = 0

    self.permute_inverse = None
    self.error_map_fct = None

    self.order = 2   # make sure other layers are up-to-date when this is called
    if mode == 'sum':
      self.set_additive_weights(float(factor)*self.N/self.source.N)
    elif mode == 'pushpull':
      self.set_pushpull_weights(float(factor)*self.N/self.source.N)
    elif mode == 'zero':
      self.W = np.zeros((self.N, self.source.N))
    else:
      raise Exception("Unknown controller mode, " + str(mode) + ".")

    self.nonlinearity = None

  def set_additive_weights(self, factor):
    """ See `__init__`, `mode == 'sum'`. """
    srcN = self.source.N
    self.W = np. zeros((self.N, srcN))

    n_per_muscle = srcN/self.N
    assignment = np.arange(srcN)/n_per_muscle

    for i in xrange(self.N):
      self.W[i, assignment==i] = factor

  def set_pushpull_weights(self, factor):
    """ See `__init__`, `mode == 'pushpull'`. """
    srcN = self.source.N
    self.W = np. zeros((self.N, srcN))

    n_per_muscle = srcN/self.N
    assignment = np.arange(srcN)/n_per_muscle

    for i in xrange(self.N):
      crt_mask = (assignment == i)
      crt_values = factor*np.ones(np.sum(crt_mask))
      crt_values[len(crt_values)/2:] *= -1
      self.W[i, crt_mask] = crt_values

  def get_motor_error(self):
    """ Find the difference between the last output and the target. """
    if self.i < self.target.shape[1]:
      target = self.target[:, self.i]
    else:
      target = np.zeros(self.N)

    return self.out - target

  def get_source_error(self):
    """ Find the difference between the last output and the target, projected
    onto the source neurons. """
    err = self.get_motor_error()
    if self.error_map_fct is not None:
      err = self.error_map_fct(err)

    res0 = np.dot(err, self.W)
    if self.permute_inverse is None:
      return res0
    else:
      return res0[self.permute_inverse]

  def set_random_permute_inverse(self, rho, subdivide_by=1):
    """ Generate a random `permute_inverse` vector that mis-assigns a fraction
    `rho` of the output neurons.

    Mis-assignment here means that first the source neurons are split into
    `self.N` groups of equal size. Then a fraction `rho` in each group is
    chosen, and `permute_inverse` is set such that these subsets are randomly
    assigned to one of the other groups. All the other neurons are properly
    assigned to their groups.

    If `subdivide_by` is different from 1, then each muscle group is further
    subdivided into the given number of subgroups. Then these subgroups are
    shuffled as before.
    """
    if rho < 1e-6:
      self.permutation = None
      return

    n_channels = subdivide_by*self.N
    m = self.source.N / n_channels      # input neurons per output
    mm = int(np.round(rho*m))           # mismatched neurons

    def derangement(v):
      # generate a "derangement" of v -- a permutation with no fixed points
      # (unless of course the length of v is 1)
      v = np.array(v, copy=True)
      n = len(v)
      if n < 2:
        return v

      while True:
        perm = np.arange(n)
        for j in range(n-1, 0, -1):
          p = np.random.randint(0, j)
          if perm[p] == j: # we want no fixed points!
            break
          else:
            perm[j], perm[p] = perm[p], perm[j]
        else:
          if perm[0] != 0:
            return v[perm]

    permutation = np.arange(self.source.N)

    mismatch_sel = []
    for channel in xrange(n_channels):
      # select which positions to mismatch for each channel
      mismatch_sel.append(channel*m + np.random.choice(m, mm, replace=False))

    # permute
    for i in xrange(mm):
      orig = [_[i] for _ in mismatch_sel]
      final = derangement(orig)
      permutation[orig] = permutation[final]

    self.permute_inverse = permutation

  def prepare(self, tmax, dt):
    if np.size(self.bias) == 1:
      self.out0 = self.bias*np.ones(self.N)
    else:
      self.out0 = np.asarray(self.bias)
    self.i = 0

    if self.nonlinearity is None:
      self.out = self.out0
    else:
      self.out = self.nonlinearity(self.out0)

  def evolve(self, t, dt):
    self.i = int_r(t/dt)
    inp = self.bias + np.dot(self.W, self.source.out)
    if self.tau is not None and self.tau > 0:
      self.out0 += (inp - self.out0)*(dt/self.tau)
    else:
      self.out0 = inp

    if self.nonlinearity is None:
      self.out = self.out0
    else:
      self.out = self.nonlinearity(self.out0)

class TwoExponentialsPlasticity(object):

  """ A heterosynaptic plasticity rule that involves two timescales.

  Attributes
  ----------
    synapses: synapses object, or tuple (source, target, W)
      This identifies the synaptic connections that are to be updated by the
      rule. If an object, it should have `source` and `target` members
      identifying the source and target layers, as well as a `W` matrix
      identifying the synaptic weights. The `synapses.source` layer should have
      an `out` field.
    tutor: neural layer
      This identifies the layer used for guiding the plasticity. This should
      have an `out` field.
    alpha, beta: float
      Parameters controlling the relative importance of the two exponential
      kernels. The kernel for smoothing the conductor signal is (for t > 0):
        alpha*exp(-t/tau1) - beta*exp(-t/tau2)       [note the minus sign!]
    tau1, tau2: float
      Timescales used for the two components of the conductor smoothing kernel
      (see above).
    theta: float
      Threshold used for tutor signal.
    rate: float
      Rate constant controlling magnitude of plasticity.
    constrain_positive: bool
      If `True`, the synaptic weights are clipped to positive values.
  """

  def __init__(self, synapses, tutor, alpha=1, beta=0,
               tau1=80.0, tau2=40.0, theta=80.0, rate=1e-10,
               constrain_positive=True):
    """ Initialize the plasticity rule.

    Arguments
    ---------
      synapses: synapses object
        This identifies the synaptic connections that are to be updated by the
        rule. This object should have `source` and `target` members identifying
        the source and target layers, as well as a `W` matrix identifying the
        synaptic weights. The `synapses.source` layer should have an `out`
        field.
      tutor: neural layer
        This identifies the layer used for guiding the plasticity. This should
        have an `out` field.
      alpha, beta: float
        Parameters controlling the relative importance of the two exponential
        kernels. The kernel for smoothing the conductor signal is (for t > 0):
          alpha/tau1*exp(-t/tau1) - beta/tau2*exp(-t/tau2)
          [note the minus sign!]
      tau1, tau2: float
        Timescales used for the two components of the conductor smoothing kernel
        (see above).
      theta: float
        Threshold used for tutor signal.
      rate: float
        Rate constant controlling magnitude of plasticity.
      constrain_positive: bool
        If `True`, the synaptic weights are clipped to positive values.
    """
    if not isinstance(synapses, (list, tuple)):
      self.synapses = synapses
    else:
      class MockSynapses(object):
        def __init__(self, src, tgt, W):
          self.source = src
          self.target = tgt
          self.W = W

      self.synapses = MockSynapses(*synapses)

    self.conductor = self.synapses.source
    self.student = self.synapses.target

    self.tutor = tutor

    self.alpha = alpha
    self.beta = beta
    self.tau1 = tau1
    self.tau2 = tau2
    self.theta = theta

    self.rate = rate

    self.constrain_positive = constrain_positive

    # make sure everything, including tutor rules, are in by the time the weight
    # update happens
    self.order = 4

  def prepare(self, tmax, dt):
    self._cond1 = np.zeros(self.conductor.N)
    self._cond2 = np.zeros(self.conductor.N)

  def evolve(self, t, dt):
    self._cond1 += (self.conductor.out - self._cond1)*(dt/self.tau1)
    self._cond2 += (self.conductor.out - self._cond2)*(dt/self.tau2)

    cond_signal = self.alpha*self._cond1 - self.beta*self._cond2
    tut_signal = self.tutor.out - self.theta

    change = self.rate*np.outer(tut_signal, cond_signal)

    w = self.synapses.W

    w += change
    if self.constrain_positive:
      w.clip(0, out=w)

class SuperExponentialPlasticity(object):

  """ A heterosynaptic plasticity rule that involves an exponential kernel and a
  t*e^{-t/\tau} kernel.

  Attributes
  ----------
    synapses: synapses object, or tuple (source, target, W)
      This identifies the synaptic connections that are to be updated by the
      rule. If an object, it should have `source` and `target` members
      identifying the source and target layers, as well as a `W` matrix
      identifying the synaptic weights. The `synapses.source` layer should have
      an `out` field.
    tutor: neural layer
      This identifies the layer used for guiding the plasticity. This should
      have an `out` field.
    alpha, beta: float
      Parameters controlling the relative importance of the two exponential
      kernels. The kernel for smoothing the conductor signal is (for t > 0):
        alpha*t/tau1**2*exp(-t/tau1) - beta/tau2*exp(-t/tau2)
        [note the minus sign!]
    tau1, tau2: float
      Timescales used for the two components of the conductor smoothing kernel
      (see above).
    theta: float
      Threshold used for tutor signal.
    rate: float
      Rate constant controlling magnitude of plasticity.
    constrain_positive: bool
      If `True`, the synaptic weights are clipped to positive values.
  """

  def __init__(self, synapses, tutor, alpha=1, beta=0,
               tau1=80.0, tau2=40.0, theta=80.0, rate=1e-10,
               constrain_positive=True):
    """ Initialize the plasticity rule.

    Arguments
    ---------
      synapses: synapses object
        This identifies the synaptic connections that are to be updated by the
        rule. This object should have `source` and `target` members identifying
        the source and target layers, as well as a `W` matrix identifying the
        synaptic weights. The `synapses.source` layer should have an `out`
        field.
      tutor: neural layer
        This identifies the layer used for guiding the plasticity. This should
        have an `out` field.
      alpha, beta: float
        Parameters controlling the relative importance of the two exponential
        kernels. The kernel for smoothing the conductor signal is (for t > 0):
          alpha*exp(-t/tau1) - beta*exp(-t/tau2)       [note the minus sign!]
      tau1, tau2: float
        Timescales used for the two components of the conductor smoothing kernel
        (see above).
      theta: float
        Threshold used for tutor signal.
      rate: float
        Rate constant controlling magnitude of plasticity.
      constrain_positive: bool
        If `True`, the synaptic weights are clipped to positive values.
    """
    if not isinstance(synapses, (list, tuple)):
      self.synapses = synapses
    else:
      class MockSynapses(object):
        def __init__(self, src, tgt, W):
          self.source = src
          self.target = tgt
          self.W = W

      self.synapses = MockSynapses(*synapses)

    self.conductor = self.synapses.source
    self.student = self.synapses.target

    self.tutor = tutor

    self.alpha = alpha
    self.beta = beta
    self.tau1 = tau1
    self.tau2 = tau2
    self.theta = theta

    self.rate = rate

    self.constrain_positive = constrain_positive

    # make sure everything, including tutor rules, are in by the time the weight
    # update happens
    self.order = 4

  def prepare(self, tmax, dt):
    self._cond1a = np.zeros(self.conductor.N)
    self._cond1 = np.zeros(self.conductor.N)
    self._cond2 = np.zeros(self.conductor.N)

  def evolve(self, t, dt):
    self._cond1a += (self.conductor.out - self._cond1)*(dt/self.tau1)
    self._cond1 += (self._cond1a - self._cond1)*(dt/self.tau1)
    self._cond2 += (self.conductor.out - self._cond2)*(dt/self.tau2)

    cond_signal = self.alpha*self._cond1 - self.beta*self._cond2
    tut_signal = self.tutor.out - self.theta

    change = self.rate*np.outer(tut_signal, cond_signal)

    w = self.synapses.W

    w += change
    if self.constrain_positive:
      w.clip(0, out=w)

class BlackboxTutorRule(object):

  """ This object sets the firing rates for the tutor neurons according to a
  rule based on an integral of the motor error.

  The calculation involves the integral
    \\int_0^t \\epsilon(t') \\frac 1 {\\tau} e^{-(t-t')/\\tau} dt',
  where \\epsilon is the motor error.

  Attributes
  ---------
    motor: motor controller
      The motor controller whose output needs to be learned.
    min_rate, max_rate: float
      Range of admissible firing rates. This is enforced by passing the
      integrated motor error through a suitable `tanh` function, provided
      `compress_rates` is `True`. If `compress_rates` is `False`, this sets the
      values to which integrals of -1 and 1 (after multiplication by `gain`),
      respectively, are mapped.
    tau: float
      Timescale over which the tutor keeps track of motor error history.
    tau_deconv1, tau_deconv2: float
      If nonzero, the tutor deconvolves the integrated motor error first with an
      exponential of timescale `tau_deconv1`, and then with one of timescale
      `tau_deconv2`.
    gain: float
      Prefactor of tutor integral *before* the `tanh` nonlinearity. If
      `compress_rates` is False, there is no `tanh`, and then the gain simply
      redefines the meaning of `min_rate` and `max_rate`.
    compress_rates: bool
      If `True`, a `tanh` nonlinearity is used to compress all firing rates to
      the range `min_rate`, `max_rate`. If `False`, only linear scaling and
      shifting is used to map an integrated motor error of 1 (after
      multiplication by `gain`) to `max_rate`; and -1 to `min_rate`.
    relaxation: float
      If not `None`, relax the rates towards `(min_rate + max_rate)/2` for a
      time starting at `tmax - relaxation` for a time `relaxation/2`, and then
      keep the rates at that value for the remaining of the simulation.
  """

  def __init__(self, motor, min_rate=40.0, max_rate=120.0, tau=np.inf,
               tau_deconv1=0, tau_deconv2=0, gain=0.001,
               compress_rates=False):
    """ Initialize the tutor rule.

    Arguments
    ---------
      motor: motor controller
        The motor controller whose output needs to be learned. This should have
        members:
          - `N` -- number of output channels
          - `source` -- the layer whose output drives the controller
                        (all that's used from `source` is `source.N`, the number
                         of neurons in that layer; this must match the size of
                         the output returned by `get_source_error`)
          - `get_source_error` -- a function that returns an estimate of the
                                  current errors in the activities of all the
                                  `motor.source` neurons
      min_rate, max_rate: float
        Range of admissible firing rates. This is enforced by passing the
        integrated motor error through a suitable `tanh` function, provided
        `compress_rates` is `True`. If `compress_rates` is `False`, this sets
        the values to which integrals of -1 and 1 (after multiplication by
        `gain`), respectively, are mapped.
      tau: float
        Timescale over which the tutor keeps track of motor error history.
      tau_deconv1, tau_deconv2: float
        If nonzero, the tutor deconvolves the integrated motor error first with
        an exponential of timescale `tau_deconv1`, and then with one of
        timescale `tau_deconv2`.
      gain: float
        Prefactor of tutor integral *before* the `tanh` nonlinearity. If
        `compress_rates` is False, there is no `tanh`, and then the gain simply
        redefines the meaning of `min_rate` and `max_rate`.
      compress_rates: bool
        If `True`, a `tanh` nonlinearity is used to compress all firing rates to
        the range `min_rate`, `max_rate`. If `False`, only linear scaling and
        shifting is used to map an integrated motor error of 1 (after
        multiplication by `gain`) to `max_rate`; and -1 to `min_rate`.
    """
    self.source = motor.source
    self.motor = motor
    self.min_rate = min_rate
    self.max_rate = max_rate
    self.tau = tau
    self.tau_deconv1 = tau_deconv1
    self.tau_deconv2 = tau_deconv2
    self.gain = gain
    self.compress_rates = compress_rates
    self.relaxation = None

    self.N = self.source.N
    
    self.order = 3

  def prepare(self, tmax, dt):
    self.integral = np.zeros(self.N)
    self.out = np.zeros(self.N)
    self.old_signal = None
    self._tmax = tmax

  def evolve(self, t, dt):
    signal = (-self.gain)*self.motor.get_source_error()
    if self.tau > 0:
      # XXX this just vanishes for infinite timescale...
      self.integral += (signal - self.integral)*(float(dt)/self.tau)
    else:
      self.integral = signal

    tau_deconv1 = self.tau_deconv1
    tau_deconv2 = self.tau_deconv2

    if tau_deconv1 < 1e-6:
      tau_deconv1 = None
    if tau_deconv2 < 1e-6:
      tau_deconv2 = None
    if tau_deconv1 is None and tau_deconv2 is not None:
      tau_deconv1, tau_deconv2 = tau_deconv2, tau_deconv1

    if tau_deconv1 is not None and tau_deconv2 is not None:
      tau_prod = float(tau_deconv1*tau_deconv2)
      tau_sum = float(tau_deconv1 + tau_deconv2)
      if self.old_signal is not None:
        derivative = (signal - self.old_signal)/dt
      else:
        derivative = np.zeros(signal.shape)

      raw_output = (tau_prod/self.tau)*derivative + \
        (tau_sum/self.tau - tau_prod/self.tau**2)*signal + \
        (1 - float(tau_deconv1)/self.tau)*(1 - float(tau_deconv2)/self.tau)* \
          self.integral

      self.old_signal = signal
    elif tau_deconv1 is not None:
      raw_output = float(tau_deconv1)/self.tau*signal + \
        (1.0 - float(tau_deconv1)/self.tau)*self.integral
    else:
      raw_output = self.integral

    mean_rate = 0.5*(self.min_rate + self.max_rate)
    half_range = 0.5*(self.max_rate - self.min_rate)
    if self.compress_rates:
      compressed = np.tanh(raw_output)
    else:
      compressed = raw_output

    if self.relaxation is not None and self.relaxation > 0:
      trel_start = self._tmax - self.relaxation
      trel = t - trel_start
      if trel >= 0:
        if trel > self.relaxation/2:
          compressed = np.zeros(len(compressed))
        else:
          xx = 1.0 - 2.0*trel/self.relaxation
          factor = xx*xx*(3.0 - 2.0*xx)
          compressed = factor*compressed

    self.out = mean_rate + half_range*compressed

class ReinforcementTutorRule(object):
  
  """ A tutor rule that uses reinforcement learning.

  The calculation involves the integral
    \\int_0^t \\R(t') \\delta g(t') \\frac 1 {\\tau} e^{-(t-t')/\\tau} dt',
  where \\R is the reward signal, and \\delta g are the fluctuations in the
  tutor rate.

  Attributes
  ---------
    tutor: tutor layer
      The tutor layer being controlled. Should have an `out` field which is
      compared to the intended firing rate to calculate the fluctuations
      `\\delta g` (see above).
    reward_src: source of reward signal
      This should be a callable (i.e., implement the `__call__` method) that
      returns the current reward signal.
    min_rate, max_rate: float
      If `constrain_rates` is `True`, this gives the range of admissible firing
      rates. Changes in the firing rate are clamped to this range.
    ini_rate: float
      Initial firing rate (Hz).
    tau: float
      Timescale over which the tutor integrates (see above).
    learning_rate: float
      Learning rate for the reinforcement updates.
    constrain_rates: bool
      If `True`, the firing rates are constrained between `min_rate` and
      `max_rate`.
    relaxation: float
      If not `None`, relax the rates towards `(min_rate + max_rate)/2` for a
      time starting at `tmax - relaxation` for a time `relaxation/2`, and then
      keep the rates at that value for the remaining of the simulation.
  """

  def __init__(self, tutor, reward_src, min_rate=40.0, max_rate=120.0,
      ini_rate=80.0, tau=0.0, learning_rate=1.0, constrain_rates=True,
      use_tutor_baseline=True, baseline_n=5, relaxation=None):
    """ Initialize object. See class docstring for explanation of arguments. """
    self.tutor = tutor
    self.reward_src = reward_src

    self.min_rate = min_rate
    self.max_rate = max_rate
    self.ini_rate = ini_rate

    self.tau = tau
    self.learning_rate = learning_rate
    self.constrain_rates = constrain_rates

    self.use_tutor_baseline = use_tutor_baseline
    self.baseline_n = baseline_n

    self.relaxation = relaxation

    self.order = 3

    self.reset_rates()
    self.reset_baseline()

  def reset_rates(self, new_rate=None):
    """ Reset the firing rates to the initial rate set at construction, or to
    `new_rate`, if provided.
    """
    if new_rate is not None:
      self.ini_rate = new_rate

    self.rates = self.ini_rate

  def reset_baseline(self):
    """ Reset the baseline that indicates the tutor output averaged over the
    last few runs when `use_tutor_baseline` is `True`.
    """
    self.baseline = None

  def prepare(self, tmax, dt):
    self.N = self.tutor.N

    nsteps = int_r(tmax/dt)

    if np.size(self.rates) == 1:
      self.rates = self.rates*np.ones((nsteps, self.N))

    if self.tau > 0:
      self.integral = np.zeros((nsteps, self.N))

    if self.use_tutor_baseline and self.baseline is None:
      self.baseline = np.copy(self.rates)

    self.out = np.copy(self.rates[0])
    self._tmax = tmax

  def evolve(self, t, dt):
    i = int_r(t/dt)

    reward = self.reward_src()
    tut_out = self.tutor.out

    if self.use_tutor_baseline:
      baseline = self.baseline[i]
    else:
      baseline = self.rates[i]

    fluctuation = (tut_out - baseline)

    learning_rate = self.learning_rate
    if self.relaxation is not None and self.relaxation > 0:
      trel_start = self._tmax - self.relaxation
      trel = t - trel_start
      if trel >= 0:
        if trel > self.relaxation/2:
          factor = 0
        else:
          xx = 1.0 - 2.0*trel/self.relaxation
          factor = xx*xx*(3.0 - 2.0*xx)
      
        learning_rate *= factor

    if self.tau == 0:
      self.rates[i] += learning_rate*reward*fluctuation
    else:
      if i > 0:
        self.integral[i, :] = (1.0 - float(dt)/self.tau)*self.integral[i-1]
      else:
        self.integral[i, :] = 0

      self.integral[i] += (dt*float(reward)/self.tau)*fluctuation
      self.rates[i] += learning_rate*self.integral[i]

    if self.constrain_rates:
      np.clip(self.rates[i], self.min_rate, self.max_rate, out=self.rates[i])

#   if self.relaxation is not None and self.relaxation > 0:
#     trel_start = self._tmax - self.relaxation
#     trel = t - trel_start
#     if trel >= 0:
#       if trel > self.relaxation/2:
#         factor = 0
#       else:
#         xx = 1.0 - 2.0*trel/self.relaxation
#         factor = xx*xx*(3.0 - 2.0*xx)
#     
#       avg_rate = 0.5*(self.min_rate + self.max_rate)
#       self.rates[i] = avg_rate + factor*(self.rates[i] - avg_rate)

    if self.use_tutor_baseline:
      self.baseline[i] += (tut_out - self.baseline[i])/float(self.baseline_n)

    self.out = np.copy(self.rates[i])

class RateLayer(object):

  """ A rate-based neural layer that combines input from several sources.

  Attributes
  ----------
    N: int
      Number of neurons in the layer.
    sources: sequence
      Sequence of source layers. Each should have an `out` field.
    Ws: sequence where elements can be
          * matrices (size `(self.N, self.sources[i].N)`), or
          * vectors (size `self.N`)
      Sequence of connection weight matrices. When an element is a vector, 
      connections are 1-to-1. The corresponding source should therefore have the
      same number of units as `self`.
    nonlinearity: callable, or None
      A function that takes the vector of linear activations (products between
      the matrices W and the outputs from the sources) and returns the layer's
      output. Set to `None` for the identity function.
    bias: array
      Biases to be used for the neurons.
  """

  def __init__(self, N, nonlinearity=None):
    """ Initialize the layer.

    Arguments
    ---------
      N: int
        Number of neurons in the layer.
      nonlinearity: callable, or None
        A function that takes the vector of linear activations (products between
        the matrices W and the outputs from the sources) and returns the layer's
        output. Set to `None` for the identity function.
    """
    self.N = N
    self.sources = []
    self.Ws = []
    self.nonlinearity = nonlinearity
    self.bias = 0.0

  def add_source(self, source):
    """ Adds a new source, setting the couplings to 0. """
    self.sources.append(source)
    self.Ws.append(0)
  
  def evolve(self, t, dt):
    def applyW(W, v):
      if np.ndim(W) <= 1:
        return np.asarray(W)*v
      else:
        return np.dot(W, v)

    if np.size(self.bias) == 1:
      res = np.repeat(self.bias, self.N)
    else:
      res = np.asarray(self.bias)

    for i in xrange(len(self.Ws)):
      res += applyW(self.Ws[i], self.sources[i].out)

    if self.nonlinearity is None:
      self.out = res
    else:
      self.out = self.nonlinearity(res)

class TableLayer(object):

  """ A layer that sets its output according to data from a matrix. """

  def __init__(self, table):
    """ Initialize the layer with the given output table. """
    self.N = len(table)
    self.table = table

  def evolve(self, t, dt):
    if self.table is not None:
      i = int_r(t/dt)
      if i < np.shape(self.table)[1]:
        self.out = 1.0*self.table[:, i]
      else:
        self.out = np.zeros(self.N)

class RateHVCLayer(object):

  """ A rate-based HVC-like layer in which neurons fire a single burst of
  constant activity at a certain time in the program.

  The amplitude of the bursts is 1. The output is stored in an attribute called
  `out`.
  """

  def __init__(self, N, burst_tmax=None, burst_length=None):
    """ Initialize the layer with `N` neurons.

    If `burst_tmax` is provided, the neurons will fire up to the time
    `burst_tmax` as opposed to the full length of the simulation.

    If `burst_length` is provided, it gives the length of each neuron's burst.
    Otherwise the length is set to completely cover the whole bursting duration.
    """
    self.N = N
    self.burst_length = burst_length
    self.burst_noise = 0.6 # ms (amount of noise in burst start times)
    self.spike_table = None
    self.burst_tmax = burst_tmax

  def prepare(self, tmax, dt):
    """ Generate the burst pattern for a new run. Distribute the bursts
    uniformly over the time of the simulation.
    """
    burst_tmax = self.burst_tmax
    if burst_tmax is None:
      burst_tmax = tmax

    # total duration of burst
    if self.burst_length is not None:
      burst_length = self.burst_length
    else:
      burst_length = burst_tmax/self.N

    # figure out when the bursts should be
    burst_start_times = np.linspace(0.0, burst_tmax - burst_length, self.N) + \
        self.burst_noise*(np.random.rand(self.N) - 0.5)

    # figure out the boolean spike table
    nsteps = int_r(burst_tmax/dt)
    self.spike_table = np.zeros((nsteps, self.N), dtype=bool)
    for i in xrange(self.N):
      start_idx = int_r(burst_start_times[i]/dt)
      if start_idx < 0:
        start_idx = 0
      end_idx = int_r((burst_start_times[i] + burst_length)/dt)
      if end_idx < 0:
        end_idx = 0
      self.spike_table[start_idx:end_idx, i] = True

  def evolve(self, t, dt):
    if self.spike_table is not None:
      i = int_r(t/dt)
      if i < len(self.spike_table):
        self.out = 1.0*self.spike_table[i]
      else:
        self.out = np.zeros(self.N)

class Connector(object):
  
  """ A class that can be used to copy data over from one simulation object to
  another.
  """

  def __init__(self, source, src_field, target, tgt_field, order=25):
    """ Set up an object that copies a field from the source to the target.

    Arguments
    ----------
      source:
      target:
        Source and target layers.
      src_field:
      tgt_field: str
        Names of fields in the source and target, respectively. This essentially
        does `setattr(target, tgt_field, getattr(source, src_field))` on every
        `evolve` call.
      order: numeric
        Set the initial execution order of this layer.
    """
    self.source = source
    self.src_field = src_field

    self.target = target
    self.tgt_field = tgt_field

    self.order = order

  def evolve(self, t, dt):
    setattr(self.target, self.tgt_field,
        getattr(self.source, self.src_field))

class MotorErrorTracker(object):
    def __init__(self, target):
        self.target = target
        self.order = 10
    
    def prepare(self, tmax, dt):
        nsteps = int(np.round(tmax/dt))
        self.motor_error = np.zeros((self.target.source.N, nsteps))
        self.overall_error = np.zeros(nsteps)
        self.t = np.zeros(nsteps)
        
        template_len = self.target.target.shape[1]
        self.overall_error = np.zeros(template_len)
    
    def evolve(self, t, dt):
        i = int(np.round(t/dt))
        self.motor_error[:, i] = self.target.get_source_error()
        template_len = len(self.overall_error)
        if i < template_len:
            self.overall_error[i] = np.linalg.norm(self.target.get_motor_error())/self.target.N
        self.t[i] = t

class SpikingLearningSimulation(object):
    
  """ A class that runs the spiking simulation for several learning cycles. """
    
  def __init__(self, target, tmax, dt, n_conductor, n_student_per_output,
               relaxation=400.0, relaxation_conductor=25.0,
               tracker_generator=None, snapshot_generator=None,
               conductor_rate_during_burst=650.0,
               conductor_spikes_per_burst=5,
               controller_mode='sum', controller_tau=25.0,
               controller_mismatch_type='random', controller_mismatch_amount=0,
               controller_mismatch_subdivide_by=1, controller_scale=0.275,
               controller_nonlinearity=None,
               student_i_external=0, student_g_inh=0,
               student_vR=-70.0, student_v_th=-55.0, student_R=260.0,
               student_tau_m=15.0, student_tau_ampa=5.0, student_tau_nmda=100.0,
               student_tau_ref=1.0,
               tutor_tau_out=5.0,
               tutor_rule_type='blackbox', tutor_rule_tau=0.0,
               tutor_rule_gain=None, tutor_rule_gain_per_student=0.5,
               tutor_rule_compress_rates=True,
               tutor_rule_min_rate=0.0, tutor_rule_max_rate=160.0,
               tutor_rule_error_tau=1000.0, tutor_rule_learning_rate=0.001,
               tutor_rule_relaxation='auto', tutor_rule_use_tutor_baseline=True,
               cs_weights_type='lognormal', cs_weights_params=(-3.57, 0.54),
               cs_weights_scale=1.0, cs_weights_fraction=1.0,
               ts_weights=0.02,
               plasticity_learning_rate=0.002, plasticity_params=(1.0, 0.0),
               plasticity_taus=(80.0, 40.0),
               plasticity_constrain_positive=True,
               progress_indicator=None):
    """ Run the simulation for several learning cycles.

    Arguments
    ---------
      target: array (shape (Nmuscles, Nsteps))
          Target output program.
      tmax:
      dt: float
          Length and granularity of target program. `tmax` should be equal to
          `Nsteps * dt`, where `Nsteps` is the number of columns of the `target`
          (see above).
      n_conductor: int
          Number of conductor neurons.
      n_student_per_output: int
          Number of student neurons per output channel. If `controller_mode` is
          not 'pushpull', the actual number of student neurons is
          `n_student_per_output * Nmuscles`, where `Nmuscles` is the number of
          rows of `target` (see above). If `controller_mode` is `pushpull`, this
          is further multiplied by 2.
      relaxation: float
          Length of time that the simulation runs past the end of the `target`.
          This ensures that all the contributions from the plasticity rule are
          considered.
      relaxation_conductor: float
          Length of time that the conductor fires past the end of the `target`.
          This is to avoid excessive decay at the end of the program.
      tracker_generator: callable
          This function is called before every simulation run with the signature
              `tracker_generator(simulator, i n)`
          where `simulator` is the object running the simulations (i.e.,
          `self`), `i` is the index of the current learning cycle, and `n` is
          the total number of learning cycles that will be simulated. The
          function should return a dictionary of objects such as `StateMonitor`s
          and `EventMonitor`s that track the system during the simulation. These
          objects will be returned in the results output structure after the run
          (see the `run` method).
      snapshot_generator: callable
          This can be a function or a pair of functions. If it is a single
          function, it is called before every simulation run with the signature
              `snapshot_generator(simulator, i n)`
          where `simulator` is the object running the simulations (i.e.,
          `self`), `i` is the index of the current learning cycle, and `n` is
          the total number of learning cycles that will be simulated. The
          function should return a dictionary that will be appended directly to
          the results output structure after the run (see the `run` method).
          This can be used to make snapshots of various structures, such as the
          conductor--student weights, during learning.
          
          When this is a pair, both elements should be functions with the same
          signature as shown above (or `None`). The first will be called before
          the simulation run, and the second after.
      conductor_rate_during_burst: float
          Conductor spike rate during each of the bursts (Hz).
      conductor_spikes_per_burst: int
          Number of conductor spikes per burst.
      controller_mode: str
          The way in which the student--output weights should be initialized.
          This can be 'sum' or 'pushpull' (see `LinearController.__init__` for
          details).
      controller_tau: float
          Timescale for smoothing of output.
      controller_scale: float
          Set the scaling for the student--output weights.
      controller_mismatch_type: str
          Method used to simulate credit assignment mismatch. Only possible
          option for now is 'random'.
      controller_mismatch_amount: float
          Fraction of student neurons whose output assignment is mismatched when
          the motor error calculation is performed (this is used by the blackbox
          tutor rule).
      controller_mismatch_subdivide_by: int
          Number of subdivisions for each controller channel when performing the
          random mismatch. Assignments between different subgroups can get
          shuffled as if they belonged to different outputs.
      controller_nonlinearty: None, or function
         If not `None`, use a (nonlinear) function to map the weighted input to
         the output. This can be used to implement linear-nonlinear controllers
         (see `LinearController`).
      student_i_external: float
          Amount of constant external current entering the student neurons.
      student_vR: float
          Set the reset potential for student neurons.
      student_v_th: float
          Set the threshold potential for student neurons.
      student_R: float
          Set the membrane resistance for student neurons.
      student_tau_m: float
          Set the membrane time constant for student neurons.
      student_tau_ampa: float
          Set the AMPAR time constant for student neurons.
      student_tau_nmda: float
          Set the NMDAR time constant for student neurons.
      student_tau_ref: float
          Set the refractory period for student neurons.
      tutor_tau_out: float
          Smoothing timescale for tutor output.
      tutor_rule_type: str
          Type of tutor rule to use. This can be 'blackbox' or 'reinforcement'.
      tutor_rule_tau: float
          Integration timescale for tutor signal (see `BlackboxTutorRule`).
      tutor_rule_gain: float, or `None`
          If not `None`, sets the gain for the blackbox tutor rule (see
          `BlackboxTutorRule`). Either this or `tutor_rule_gain_per_student`
          should be non-`None`. 
      tutor_rule_gain_per_student: float, or `None`
          If not `None`, the gain for the blackbox tutor rule (see
          `BlackboxTutorRule`) is set proportional to the number of student
          neurons per output channel, `n_student_per_channel`.
      tutor_rule_compress_rates: bool
          Sets the `compress_rates` option for the blacktox tutor rule (see
          `BlackboxTutorRule`).
      tutor_rule_min_rate: float
      tutor_rule_max_rate: float
          Sets the minimum and maximum rate for the tutor rule (see
          `BlackboxTutorRule`).
      tutor_rule_error_tau: float
          Timescale used by the reinforcement rule to calculate the baseline for
          the reward signal Set to infinity to avoid subtracting a baseline.
          (see `ReinforcementTutorRule`)
      tutor_rule_learning_rate: float
          Learning rate for the reinforcement rule.
      tutor_rule_relaxation: None, float, or 'auto'
          Relaxation time for tutor rule (see `BlackboxTutorRule` and
          `ReinforcementTutorRule`). If set to 'auto', the value from
          `relaxation` (see above) is used.  If set to `None` or 0, the tutor
          rule is active until the end of the simulation. Otherwise the given
          relaxation time is used.
      tutor_rule_use_tutor_baseline: bool
          If `True`, the baseline used to calculate tutor fluctuations in the
          reinforcement rule is estimated by averaging actual tutor rates over
          the last few learning cycles. If `False`, the intended firing rate is
          used as a baseline (see `ReinforcementTutorRule`).
      cs_weights_type: str
      cs_weights_params:
          Sets the way in which the conductor--student weights should be
          initialized. This can be
            'zero':       set the weights to zero
            'constant':   set all the weights equal to `cs_weights_params`
            'normal':     use Gaussian random variables, parameters (mean,
                          st.dev.) given by `cs_weights_params`
            'lognormal':  use log-normal random variables, parameters (mu,
                          sigma) given by `cs_weights_params`
      cs_weights_scale: float
          Set a scaling factor for all the conductor--student weights. This is
          applied after the weights are calculated according to
          `cs_weights_type` and `cs_weights_params`.
      cs_weights_fraction: float
          Fraction of conductor--student weights that are nonzero.
      ts_weights: float
          The value of the tutor--student synaptic strength.
      plasticity_learning_rate: float
          The learning rate of the plasticity rule (see
          `TwoExponentialsPlasticity`).
      plasticity_params: (alpha, beta)
          The parameters used by the plasticity rule (see
          `TwoExponentialsPlasticity`).
      plasticity_taus: (tau1, tau2)
          The timescales used by the plasticity rule (see
          `TwoExponentialsPlasticity`).
      plasticity_constrain_positive: bool
          Whether to keep conductor--student weights non-negative or not
          (see `TwoExponentialsPlasticity`).
      progress_indicator: None, str, or class
          If `None`, there will be no indication of progress. If a class, the
          simulation's constructor will call the class's constructor with `self`
          as an argument. The `__call__` method of the resulting object will
          then be called before every learning cycle, with arguments `(i, n)`,
          where `i` is the index of the learning cycle and `n` is the total
          number of learning cycles.
    """
    self.target = np.asarray(target)
    self.tmax = float(tmax)
    self.dt = float(dt)
    
    self.n_muscles = len(self.target)
    
    self.n_conductor = n_conductor
    self.n_student_per_output = n_student_per_output
    
    self.relaxation = relaxation
    self.relaxation_conductor = relaxation_conductor
    
    self.tracker_generator = tracker_generator
    self.snapshot_generator = snapshot_generator
    
    if not hasattr(self.snapshot_generator, '__len__'):
        self.snapshot_generator = (self.snapshot_generator, None)
    
    self.conductor_rate_during_burst = conductor_rate_during_burst
    self.conductor_spikes_per_burst = conductor_spikes_per_burst
    
    self.controller_mode = controller_mode
    self.controller_tau = controller_tau
    self.controller_scale = controller_scale
    self.controller_mismatch_type = controller_mismatch_type
    self.controller_mismatch_amount = controller_mismatch_amount
    self.controller_mismatch_subdivide_by = controller_mismatch_subdivide_by
    self.controller_nonlinearity = controller_nonlinearity

    self.student_i_external = student_i_external
    self.student_g_inh = student_g_inh
    self.student_vR = student_vR
    self.student_v_th = student_v_th
    self.student_R = student_R
    self.student_tau_m = student_tau_m
    self.student_tau_ampa = student_tau_ampa
    self.student_tau_nmda = student_tau_nmda
    self.student_tau_ref = student_tau_ref

    self.tutor_tau_out = tutor_tau_out

    self.tutor_rule_type = tutor_rule_type
    self.tutor_rule_tau = tutor_rule_tau
    self.tutor_rule_gain = tutor_rule_gain
    self.tutor_rule_gain_per_student = tutor_rule_gain_per_student
    self.tutor_rule_error_tau = tutor_rule_error_tau
    self.tutor_rule_learning_rate = tutor_rule_learning_rate
    
    self.tutor_rule_compress_rates = tutor_rule_compress_rates
    self.tutor_rule_min_rate = tutor_rule_min_rate
    self.tutor_rule_max_rate = tutor_rule_max_rate
    self.tutor_rule_relaxation = tutor_rule_relaxation
    self.tutor_rule_use_tutor_baseline = tutor_rule_use_tutor_baseline
    
    self.cs_weights_type = cs_weights_type
    self.cs_weights_params = cs_weights_params
    self.cs_weights_scale = cs_weights_scale
    self.cs_weights_fraction = cs_weights_fraction
    
    self.ts_weights = ts_weights
    
    self.plasticity_learning_rate = plasticity_learning_rate
    self.plasticity_params = plasticity_params
    self.plasticity_taus = plasticity_taus
    self.plasticity_constrain_positive = plasticity_constrain_positive
    
    self.progress_indicator = progress_indicator

    self.setup()
    
  def setup(self):
    """ Create the components of the simulation. """
    # process some of the options
    self.n_student = self.n_student_per_output*self.n_muscles
    
    if self.controller_mode == 'pushpull':
      self.n_student *= 2
    
    if self.tutor_rule_gain is None:
      self.tutor_rule_actual_gain = (self.tutor_rule_gain_per_student*
          self.n_student_per_output)
    else:
      self.tutor_rule_actual_gain = self.tutor_rule_gain
    
    self.total_time = self.tmax + self.relaxation
    self.stimes = np.arange(0, self.total_time, self.dt)
    
    self._current_res = []
    
    # build components
    self.conductor = HVCLikeLayer(self.n_conductor,
        burst_tmax=self.tmax+self.relaxation_conductor)
    self.conductor.spikes_per_burst = self.conductor_spikes_per_burst
    self.conductor.rate_during_burst = self.conductor_rate_during_burst

    self.student = StudentLayer(self.n_student,
        vR=self.student_vR, v_th=self.student_v_th, R=self.student_R,
        tau_m=self.student_tau_m, tau_ampa=self.student_tau_ampa,
        tau_nmda=self.student_tau_nmda, tau_ref=self.student_tau_ref,
        g_inh=self.student_g_inh)
    self.motor = LinearController(self.student, self.target,
        factor=self.controller_scale,
        mode=self.controller_mode, tau=self.controller_tau)
    self.motor.nonlinearity = self.controller_nonlinearity
    if self.controller_mismatch_amount > 0:
      if self.controller_mismatch_type != 'random':
        raise Exception('Unknown controller_mismatch_type ' +
                        str(self.controller_mismatch_type) + '.')
      self.motor.set_random_permute_inverse(
          self.controller_mismatch_amount,
          subdivide_by=self.controller_mismatch_subdivide_by)

    self.tutor = RandomLayer(self.n_student)
    self.tutor.tau_out = self.tutor_tau_out
    
    # tutor rule controls the rate of the tutor layer
    if self.tutor_rule_type == 'blackbox':
      self.tutor_rule = BlackboxTutorRule(self.motor, tau=self.tutor_rule_tau,
          gain=self.tutor_rule_actual_gain,
          compress_rates=self.tutor_rule_compress_rates,
          min_rate=self.tutor_rule_min_rate,
          max_rate=self.tutor_rule_max_rate)
    elif self.tutor_rule_type == 'reinforcement':
      class RewardFunction(object):
        def __init__(self, motor, tau):
          self.motor = motor
          self.tau = tau
          self.baseline = 0.0
          self.order = 2.5

        def prepare(self, tmax, dt):
          self.reward = 0.0

        def evolve(self, t, dt):
          motor_error = self.motor.get_motor_error()
          self.inst_reward = -np.linalg.norm(motor_error)

          if np.isinf(self.tau):
            self.reward = self.inst_reward
          else:
            self.reward = self.inst_reward - self.baseline
            self.baseline += (self.inst_reward - self.baseline)*(dt/self.tau)
        
        def __call__(self):
          return self.reward

      self.reward_src = RewardFunction(self.motor, self.tutor_rule_error_tau)
      self.tutor_rule = ReinforcementTutorRule(self.tutor, self.reward_src,
          tau=self.tutor_rule_tau,
          constrain_rates=self.tutor_rule_compress_rates,
          min_rate=self.tutor_rule_min_rate,
          max_rate=self.tutor_rule_max_rate,
          learning_rate=self.tutor_rule_learning_rate,
          use_tutor_baseline=self.tutor_rule_use_tutor_baseline)
    else:
      raise Exception('Unknown tutor_rule_type  (' + str(self.tutor_rule_type) +
                      ').')
    # the tutor rule will wind down its activity during relaxation time
    if self.tutor_rule_relaxation == 'auto':
      self.tutor_rule.relaxation = self.relaxation
    else:
      self.tutor_rule.relaxation = self.tutor_rule_relaxation

    class SlidingConnector(object):
      def __init__(self, source, target):
        self.source = source
        self.target = target
      
      def evolve(self, t, dt):
        i = int_r(t/dt)
        self.target.rate[i, :] = self.source.out

    self.tutor_connector = SlidingConnector(self.tutor_rule, self.tutor)

    # set inputs to student
    self.student.i_ext_init = self.student_i_external

    self.conductor_synapses = ExcitatorySynapses(self.conductor,
                                                 self.student)
    self.tutor_synapses = ExcitatorySynapses(self.tutor, self.student,
        f_nmda=0.9, one_to_one=True)

    # generate the conductor--student weights
    self.init_cs_weights()
    
    # set tutor--student weights
    self.tutor_synapses.W = self.ts_weights*np.ones(self.n_student)

    # inform the student and tutor of the necessary quantities to set the
    # initial values for i_nmda and i_ampa
    self.tutor.avg_rates = (self.tutor_rule_min_rate +
                            self.tutor_rule_max_rate)/2.0
    self.student.v_ref = (self.student.v_th + self.student.vR)/2.0
    
    # initialize the plasiticity rule
    self.plasticity = TwoExponentialsPlasticity(
        self.conductor_synapses, self.tutor,
        rate=self.plasticity_learning_rate,
        alpha=self.plasticity_params[0], beta=self.plasticity_params[1],
        tau1=self.plasticity_taus[0], tau2=self.plasticity_taus[1],
        constrain_positive=self.plasticity_constrain_positive)

  def init_cs_weights(self):
    """ Initialize conductor--student weights. """
    if self.cs_weights_type == 'zero':
      W = np.zeros((self.n_student, self.n_conductor))
    elif self.cs_weights_type == 'constant':
      W = self.cs_weights_params*np.ones((self.n_student,
                                          self.n_conductor))
    elif self.cs_weights_type == 'normal':
      W = (self.cs_weights_params[0] + self.cs_weights_params[1]*
            np.random.randn(self.n_student, self.n_conductor))
    elif self.cs_weights_type == 'lognormal':
      W = np.random.lognormal(*self.cs_weights_params,
                              size=(self.n_student, self.n_conductor))
    
    W *= self.cs_weights_scale
    W[np.random.rand(*W.shape) >= self.cs_weights_fraction] = 0

    self.conductor_synapses.W = W

  def run(self, n_runs):
    """ Run the simulation for `n_runs` learning cycles.
    
    This function intercepts `KeyboardInterrupt` exceptions and returns the
    results up to the time of the keyboard intercept.
    """
    res = []
    
    self._current_res = res

    if self.progress_indicator is not None:
      progress_indicator = self.progress_indicator(self)
    else:
      progress_indicator = None

    try:
      for i in xrange(n_runs):
        if progress_indicator is not None:
          progress_indicator(i, n_runs)
        
        # make the pre-run snapshots
        if self.snapshot_generator[0] is not None:
          snaps_pre = self.snapshot_generator[0](self, i, n_runs)
        else:
          snaps_pre = {}
        
        # get the trackers
        if self.tracker_generator is not None:
          trackers = self.tracker_generator(self, i, n_runs)
          if trackers is None:
            trackers = {}
        else:
          trackers = {}
                        
        # no matter what, we will need an error tracker to calculate average
        # error
        M_merr = MotorErrorTracker(self.motor)
        
        # create and run the simulation
        others = []
        if self.tutor_rule_type == 'reinforcement':
          others.append(self.reward_src)
        sim = simulation.Simulation(
            self.conductor, self.student, self.tutor,
            self.conductor_synapses, self.tutor_synapses,
            self.tutor_rule, self.motor, self.plasticity, self.tutor_connector,
            M_merr, *(others + trackers.values()), dt=self.dt)
        sim.run(self.total_time)
        
        # make the post-run snapshots
        if self.snapshot_generator[1] is not None:
            snaps_post = self.snapshot_generator[1](self, i, n_runs)
        else:
            snaps_post = {}
        
        crt_res = {'average_error': np.mean(M_merr.overall_error)}
        crt_res.update(snaps_pre)
        crt_res.update(snaps_post)
        crt_res.update(trackers)
        
        res.append(crt_res)
          
      if progress_indicator is not None:
        progress_indicator(n_runs, n_runs)
    except KeyboardInterrupt:
      pass
    
    return res
