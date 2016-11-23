""" Set up some helpful functions for visualization and data processing. """

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import os

from matplotlib import animation

# visualization
def show_repetition_pattern(monitors, **kwargs):
    """ Show the spiking pattern for a set of neurons over a series of repetitions.
    
    Parameters
    ----------
      monitors: event monitor, or sequence of event monitors
          An event monitor or a sequence of event monitors identifying the spiking times
          for a set of neurons. Each event monitor should have fields `t` and `i`
          identifying spike times and the indices of the neurons that spiked, respectively.
      idx: int, or sequence of int (optional)
          This can be a single integer or a sequence of integers showing which neurons
          to focus on. If idx is not provided, the unit identifiers in the event monitors
          will be ignored, as if all the events were produced by a single neuron.
      tmin
      tmax: None, or float
          Range of times where to show spikes.
      flip_y: boolean (optional)
          Whether to flip the y-axis, putting the first neuron on top.
      colors: sequence of Matplotlib color specifications
          List of colors to cycle through when showing patterns from several neurons.
          
      Other keyword arguments are directly passed to plot.
    
    Returns nothing.
    """
    # defaults
    idx = kwargs.pop('idx', None)
    
    tmin = kwargs.pop('tmin', None)
    tmax = kwargs.pop('tmax', None)
    
    flip_y = kwargs.pop('flip_y', True)
#    colors = kwargs.pop('colors', ['k', 'r', 'g', 'b', [1.0, 0.5, 0.0]])
#    colors = kwargs.pop('colors', ['b', [1.0, 0.2, 0.0], 'k',
#                                   [1.0, 0.8, 0.0], [1.0, 0.0, 0.6]])
    colors = kwargs.pop('colors',
        [[0.20, 0.23, 0.59], [0.92, 0.18, 0.18], [0.13, 0.13, 0.13],
        [0.82, 0.52, 0.18], [0.93, 0.13, 0.54]])

    # altered Matplotlib defaults
    ms = kwargs.setdefault('ms', 1.5)
    mew = kwargs.setdefault('mew', 0.6)

    # if `monitors` isn't already a list, make it so
    if hasattr(monitors, 't') and hasattr(monitors, 'i'):
        monitors = [monitors]

    # the following code is easier if `idx` is always a list
    if idx is None:
        idx = [None]
    
    # draw the neurons one by one, each in a different color
    for k, crt_idx in enumerate(idx):
        crt_color = colors[k%len(colors)]
        kwargs['color'] = crt_color
        for i, monitor in enumerate(monitors):
            crt_t = np.asarray(monitor.t)
            crt_i = np.asarray(monitor.i)
            events = crt_t[crt_i == crt_idx] if crt_idx is not None else crt_t
            yshift = i*0.8/(len(monitors)-1) - 0.4 if len(monitors) > 1 else 0.0
            plt.plot(events, k + yshift + np.zeros(np.size(events)), '|', **kwargs)
        
    # prettify the plot
    plt.xlabel('$t$ (ms)')
    plt.ylabel('Neuron index')
    plt.ylim(-1, len(idx))

    # flip the axis if needed
    if flip_y:
        plt.gca().invert_yaxis()
    
    # some settings might have been changed by Seaborn -- resetting everything to be sure
    ax = plt.gca()
    ax.grid(False)
    ax.set_axis_bgcolor('white')
    
    # make sure the box is showing
    for sp_name in ['left', 'right', 'top', 'bottom']:
        ax.spines[sp_name].set_alpha(1.0)
        ax.spines[sp_name].set_color('black')
        ax.spines[sp_name].set_linewidth(1.0)
        ax.spines[sp_name].set_visible(True)

def plot_evolution(res, target, dt, max_traces=100, show_final=True):
    """ Show final motor program and the progression of learning.
    
    Parameters
    ----------
      res: sequence of results structures
          List of results, with one entry per learning rendition. Each element needs to
          contain an attribute `motor`, an `N` x `n` matrix, where `N` is the number of
          muscles, and `n` the number of time steps.
      target: 2d array
          The `N` x `n` matrix giving the target muscle output.
      dt: float
          Simulation time step.
      max_traces: int
          Maximum number of traces to draw. If `len(res)>max_traces`, a uniformly-spaced
          subset of the results is drawn to ensure there are only about `max_traces` of
          them. The actual number can vary between `0.75*max_traces` and `1.5*max_traces`.
      show_final: bool
          If True, display the final trace of the results (the last entry that contains a
          `motor` attribute).
          
    Returns nothing.
    """
    n_muscles = len(target)
    # dimensions of plot grid -- approximately same number of plots horizontally
    # as vertically
    n_plots_x = int(math.ceil(math.sqrt(n_muscles)))
    dim_plots = (n_plots_x, int(math.ceil(float(n_muscles) / n_plots_x)))
    
    # keep the same aspect ratio for the plots as the current Matplotlib default
    default_size = plt.rcParams['figure.figsize']
    aspect_ratio = float(default_size[1])/default_size[0]
    
    # scale the plots so that the horizontal dimension of the whole grid is 15 inches
    full_x_size = 15.0
    img_x_size = full_x_size / dim_plots[0]
    img_y_size = aspect_ratio*img_x_size
    
    fig_size = [full_x_size, img_y_size*dim_plots[1]]
    plt.figure(figsize=fig_size)
    
    # y-axis labels
    names = ['Muscle {}'.format(i+1) for i in xrange(n_muscles)]
    
    target_times = dt*np.arange(target.shape[1])
    has_motor = [crt_res.has_key('motor') for crt_res in res]
    motorized_idxs = np.nonzero(has_motor)[0]
    
    for mus in xrange(n_muscles):
        plt.subplot(dim_plots[1], dim_plots[0], mus+1)
        
        # don't draw many more than `max_traces` traces
        if len(motorized_idxs) > max_traces:
            plot_step = int(round(float(len(motorized_idxs))/max_traces))
        else:
            plot_step = 1
            
        for i in xrange(0, len(motorized_idxs), plot_step):
            crt_res = res[motorized_idxs[i]]
            
            # color becomes more saturated later during the learning
            ratio = float(motorized_idxs[i] + 1)/len(res)
            color = [1.0, 0.75*(1-ratio), 0.75*(1-ratio)]
            
            plt.plot(crt_res['motor'].t, crt_res['motor'].out[mus], color=color, lw=0.5)
 
        # show the target
        plt.plot(target_times, target[mus], 'k')
        # show the final trace, if it exists
        if show_final and len(motorized_idxs) > 0:
            crt_res = res[motorized_idxs[-1]]
            plt.plot(crt_res['motor'].t, crt_res['motor'].out[mus], color=[1.0, 0.5, 0.0])

        plt.xlabel('$t$ (ms)')
        plt.ylabel(names[mus] + ' (a.u.)')

def display_error_evolution(res, key='average_error'):
    """ Display a plot of how the reproduction error evolved, emphasizing its final value.
    
    Parameters
    ----------
      res: sequence of results structures, or sequence of numbers
          List of results with one entry per learning rendition. Each element
          should have an attribute called `average_error` (or different, if
          `key` is used; or it could be the error itself; see below), which is a
          number giving an estimate of the total error during a run.
      key: string
          This gives the name of the results attribute used for the error.
          Alternatively, if this is `None`, it is assumed that `res` is a list
          of errors.
          
      Returns nothing.
    """
    # plot the trace
    if key is not None:
      error_evolution = [x[key] for x in res]
    else:
      error_evolution = res
    plt.plot(error_evolution, '.-k')
    
    # show the final value with a label
    if np.isfinite(error_evolution[-1]):
      tmp_xy = (len(res)-1, error_evolution[-1])
      yrange = (lambda x: x[1] - x[0])(plt.ylim())
      plt.annotate(s='{:0.3f}'.format(tmp_xy[1]), xy=tmp_xy,
                   xytext=(tmp_xy[0]*0.9, tmp_xy[1]+0.25*yrange),
                   fontsize='large',
                   arrowprops=dict(color='red', arrowstyle='->', lw=2))
    
    # also show the median over the last 50 renditions
    last_error = np.median(error_evolution[-50:])
    if np.isfinite(last_error):
      tmp_xy = (max(0, len(res)-100), last_error)
      plt.plot([tmp_xy[0], len(res)], 2*[last_error], '--', lw=2,
               color=[0, 0.8, 0.2])
      plt.annotate(s='{:0.3f}'.format(last_error), xy=tmp_xy,
                   xytext=(-32, -4),
                   fontsize='large',
                   textcoords='offset points',
                   backgroundcolor=[1.0, 1.0, 1.0, 0.85],
                   color=[0.0, 0.6, 0.0])

def draw_multi_traces(t, y, edge_factor=0.2, color_fct=None, lw=1, rev_y=True,
                      fill_alpha=None):
    """ Draw multiple traces on the same plot. """
    n = len(y)
    height = 1.0/n/(1.0 + edge_factor)
    label_y = []
    label_txt = []
    if color_fct is None:
        color_fct = lambda _: 'k'
    for i in xrange(n):
        yc = (i + 0.5) / n
        if rev_y:
          yc = 1 - yc
        datac = 0.5*(np.min(y[i]) + np.max(y[i]))
        datar = max(np.ptp(y[i]), np.abs(datac)/1e6)
        if datar == 0:
            datar = 1.0

        ydata = yc + (y[i] - datac)*height/datar
        y0 = yc - datac*height/datar
        if fill_alpha is not None:
          plt.fill_between(t, np.full(np.shape(ydata), y0), ydata,
                           color=color_fct(i), alpha=fill_alpha)
        plt.plot(t, ydata, c=color_fct(i), lw=lw)
        label_y.append(yc)
        label_txt.append(i+1)
    
    plt.yticks(label_y, label_txt)
    plt.ylim(0, 1)

# data conversion
def collect_isi(spikes_list, tmax=None):
    """ Collect inter-spike interval data from a list of spike monitors.
    
    Each spike monitor can collect spikes from several units. If so, the
    inter-spike intervals (ISIs) are calculated independently for each unit,
    and then concatenated in the final output. The ISIs for the different
    items in `spikes_list` are also concatenated.

    If `tmax` is not `None`, spikes at or after `tmax` are ignored.
    
    Returns
    -------
      A Numpy array of inter-spike intervals.
    """
    isi = []
    for crt_spikes in spikes_list:
        t = np.asarray(crt_spikes.t)
        idx = np.asarray(crt_spikes.i)
        for i in np.unique(idx):
            mask = (idx == i)
            if tmax is not None:
              mask &= (t < tmax)
            isi.extend(np.diff(t[mask]))
    
    return np.asarray(isi)

def select_events_by_unit(monitor, idx0, idx1):
    """ Generate an `EventMonitor`-like structure that contains the information
    for only the units in the range `idx0:idx1`.
    
    Parameters
    ----------
      monitor:
          An `EventMonitor`-like structure with fields `t` and `i`.
      idx0
      idx1: int
          The range of unit indices to keep in the return structure. The range
          includes `idx0` and excludes `idx1`.
    
    Returns
    -------
      Another `EventMonitor`-like structure that contains only the data from the
      units in the range `idx0:idx1`.
    """
    class EventLike(object):
        def __init__(self, t, i, N):
            self.t = t
            self.i = i
            self.N = N
        
        def __len__(self):
            return len(self.t)
    
    mt = np.asarray(monitor.t)
    mi = np.asarray(monitor.i)
    mask = ((mi >= idx0) & (mi < idx1))
    return EventLike(mt[mask], mi[mask], idx1 - idx0)

def show_program_development(reses, axs, channel=0, stages=[], ymax=None,
        lw=1.0, target=None, bbox_pos=(1.15, 1.05), color=[0.831, 0.333, 0]):
    """ Make a figure comparing the motor output to the target at a few stages in the learning
    process.
    
    Parameters
    ----------
      res: list of lists
          The results structure used for making the figure. Each entry should be a
          dictionary containing 'motor' output. The entries should also contain information
          about the target program, unless the `target` argument is used.
      axs: list of Matplotlib Axes
          These are the axes in which the traces should be drawn.
      channel: int
          Which output channel to draw.
      stages: sequence of int
          Which learning stages to draw.
      ymax: float
          Upper bound for y-axes.
      lw: float
          Line width.
    """
    idxs = np.asarray(stages)
    n_stages = len(idxs)
    if n_stages == 0:
        return
    if n_stages != len(axs):
        raise Exception("Number of axes should match number of stages.")
        
    if target is None:
      target = reses['target']

    try:
      # convert reses to Numpy arrays, so we can index easily
      res = np.asarray(reses['trace'])[stages]
    except TypeError:
      res = np.asarray(reses)[stages]

    times = res[0]['motor'].t[:np.shape(target)[1]]
    tmax = times[-1] + times[1] - times[0]

    # and draw each subplot
    for i, (crt_res, ax) in enumerate(zip(res, axs)):
        motor = crt_res['motor']
        
        ax.plot(times, target[channel], ':k', lw=1, label='target')
        ax.set_title('Stage {}'.format(idxs[i] + 1))
        ax.set_xlabel('time', labelpad=-5)
        
        if ymax is None:
            ylim = ax.get_ylim()[1]
        else:
            ylim = ymax
            
        ax.set_ylim(0, ylim)
        
        ax.plot(motor.t[:len(times)], motor.out[channel][:len(times)],
                color=color, lw=lw, label='output')
        ax.legend(loc='upper right', bbox_to_anchor=bbox_pos, fontsize=8)
        
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        
#        ax.xaxis.set_tick_params(labelsize=8)
#        ax.yaxis.set_tick_params(labelsize=8)
        
        ax.xaxis.set_ticks([0, tmax])
        ax.yaxis.set_ticks(np.arange(ylim+1, step=20))
        
#        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
#            item.set_fontsize(10)

def draw_convergence_plot(res, axs, target_lw=1, extra_traces=None,
        extra_colors=None, inset=False, alpha=None, beta=None, tau_tutor=None,
        legend_pos=(1.1, 1.1),
        target=None, inset_pos=[0.35, 0.35, 0.45, 0.45],
        err_color=[0.200, 0.357, 0.400], out_color=[0.831, 0.333, 0.000]):
    """ Draw the error trace in `axs[0]` and the final output in `axs[1]`.
    
    If `inset` is `True`, `axs` should be a single axis, not a set, and the
    final output will be drawn in an inset.
    """
    if inset:
      axs = [axs]

    # check if res is only the trace
    try:
      res['error_trace']

      if alpha is None:
        alpha = res['alpha']
      if beta is None:
        beta = res['beta']

      if tau_tutor is None:
        try:
          tau_tutor = res['tau_tutor']
        except KeyError:
          tau_tutor = res['tutor_rule_tau']
    except TypeError:
      res = dict(trace=res)
      res['error_trace'] = [_['average_error'] for _ in res['trace']]

    if target is None:
      target = res['target']

#    axs[0].semilogy(res['error_trace'], '-k', marker='.', lw=1, color=[0, 0.2, 0.6])
    axs[0].plot(res['error_trace'], '-k', marker='.', lw=1, color=err_color)
#    axs[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    axs[0].spines['right'].set_color('none')
    axs[0].spines['top'].set_color('none')

    axs[0].set_xlabel('repetitions')
    axs[0].set_ylabel('error')
#    axs[0].set_ylim(0.1, 15);

    axs[0].set_title(r'$\alpha={:.1f},\; \beta={:.1f},\; \tau_\mathrm{{tutor}}={:.1f}$'.format(
            alpha, beta, tau_tutor))

    if inset:
      axs.append(plt.axes(inset_pos))
    
    motor = res['trace'][-1]['motor']
    times = motor.t[:np.shape(target)[1]]

    axs[1].plot(times, target[0], ':k', lw=target_lw, label='target')
    axs[1].plot(motor.t[:len(times)], motor.out[0][:len(times)], color=out_color, label='output')

    if extra_traces is None:
        extra_traces = []
    if extra_colors is None:
        extra_colors = []

    for (i, res_i) in enumerate(extra_traces):
        crt_motor = res['trace'][res_i]['motor']
        crt_color = (out_color if len(extra_colors) == 0 else
                      extra_colors[i%len(extra_colors)])
        axs[1].plot(crt_motor.t[:len(times)], crt_motor.out[0][:len(times)],
        c=crt_color, lw=0.5)
    
    axs[1].spines['right'].set_color('none')
    axs[1].spines['top'].set_color('none')
    
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('output')
    
    axs[1].legend(loc='upper right', bbox_to_anchor=legend_pos,
        prop={'size': 8 if inset else 10})
    
    axs[1].set_ylim(0, axs[1].get_ylim()[1])

    if inset:
      axs[1].set_xticks([])

    fontsizes = [None]
    if inset:
      fontsizes.append(8)
    else:
      fontsizes.append(None)
    for i, ax in enumerate(axs):
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            if fontsizes[i] is not None:
                item.set_fontsize(fontsizes[i])

    return axs

# data structures
class Structure(object):
    
    """ This is essentially a dictionary with convenient access semantics.
    
    The default constructor makes an object that is completely empty, save for the
    _fields and _asdict attributes (see below). This structure can then be used to
    add or remove fields and modify and read their values. Thus this class
    provides convenient access to the underlying dictionary object. The advantage is
    not having to use square brackets and quotation marks for access. Of course, fields
    with spaces in them, starting with digits, or non-character fields are not accessible
    in this simple way.
    
    Attributes
    ----------
    _fields     -- property equal to a list of the fields stored in the structure.
    _asdict()   -- return a dictionary with the structure contents
    """
    def __init__(self, **kwargs):
        """ Initialize structure with the given attributes. """
        for (key, value) in kwargs.items():
            setattr(self, key, value)
    
    @property
    def _fields(self):
        return self.__dict__.keys()
    
    def _asdict(self):
        return self.__dict__

    def __repr__(self):
        return 'Storage(' + ', '.join(['\n  ' + field + '=' + repr(getattr(self, field))
                                       for field in self._fields]) + ')'

def safe_save_fig(base_name, pdf=True, svg=False, png=False):
    if pdf:
      pdf_name = base_name + '.pdf'
      if not os.path.exists(pdf_name):
          plt.savefig(pdf_name, bbox_inches='tight')
      else:
          print('NOTE: not saving PDF; already exists (' + pdf_name + ')')
        
    if svg:
      svg_name = base_name + '.svg'
      if not os.path.exists(svg_name):
          plt.savefig(svg_name, bbox_inches='tight')
      else:
          print('NOTE: not saving SVG; already exists (' + svg_name + ')')
    
    if png:
      png_name = base_name + '.png'
      if not os.path.exists(png_name):
          plt.savefig(png_name, bbox_inches='tight')
      else:
          print('NOTE: not saving PNG; already exists (' + png_name + ')')

def get_convergence_time(res, threshold=2.0):
    return len(res) - (np.asarray([x['average_error'] for x in res][::-1]) > threshold).nonzero()[0][0]

def make_heatmap_plot(res_matrix, size=(2.5, 2), vmin=0.1, vmax=5, norm='log',
        args_matrix=None, sim_idx=-1, xlabelpad=0, ylabelpad=0):
    plt.figure(figsize=size)
    
    try:
        final_errors = np.asarray([[_['error_trace'][sim_idx] for _ in crt_row] for crt_row in res_matrix])
    except IndexError:
        final_errors = np.asarray([[_[sim_idx] for _ in crt_row] for crt_row in
            res_matrix])

    final_errors[~np.isfinite(final_errors)] = np.inf

    if args_matrix is None:
        tau_levels = [_['params']['tutor_rule_tau'] for _ in res_matrix[0]]
    else:
        tau_levels = [_['tutor_rule_tau'] for _ in args_matrix[0]]
    
    if norm == 'log':
        norm = mpl.colors.LogNorm()
        
        low_tick = 10**math.floor(math.log10(vmin))
        high_tick = 10**math.ceil(math.log10(vmax))
        
        ticks = []
        crt_order = low_tick
        for i in xrange(int(np.log10(low_tick)), int(np.log10(high_tick))+1):
            ticks.extend([crt_order, 2*crt_order, 5*crt_order])
            crt_order *= 10
        
        order = low_tick
    elif norm == 'linear':
        norm = None
        
        vmid = math.sqrt(vmin**2 + vmax**2)
        order = 10**(math.floor(math.log10(vmid)) - 1)
        
        low_tick = math.floor(vmin/order)*order
        high_tick = math.ceil(vmax/order)*order
        
        ticks = np.arange(low_tick, high_tick, order)
    
    plt.imshow(np.clip(final_errors, vmin, vmax),
               interpolation='nearest', cmap='Blues', norm=norm, vmin=vmin, vmax=vmax)
    plt.grid(False)
    plt.xticks(range(len(tau_levels)), ['{:.0f}'.format(crt_tau) for crt_tau in tau_levels],
               fontsize=8)
    plt.yticks(range(len(tau_levels)), ['{:.0f}'.format(crt_tau) for crt_tau in tau_levels],
               fontsize=8)
    plt.xticks(rotation=45)
    
    cbar_format = '%.{}f'.format(max(0, -int(math.log10(order))+1))
    cbar = plt.colorbar(ticks=ticks, format=cbar_format);

    plt.xlabel(r'$\tau_\mathrm{tutor}$', labelpad=xlabelpad)
    plt.ylabel(r'$\frac {\alpha \tau_1 - \beta \tau_2} {\alpha - \beta}$',
               labelpad=ylabelpad)

def make_convergence_map(res_matrix, size=(2, 2), max_error=20, show_no_convergence=True,
                         no_convergence_color=[0.902, 0.820, 0.765, 0.5],
                         err_color=[0.200, 0.357, 0.400],
                         args_matrix=None, max_steps=None,
                         xlabelpad=0, ylabelpad=0):
    fig = plt.figure(figsize=size)
    
    if args_matrix is None:
        tau_levels = [_['params']['tutor_rule_tau'] for _ in res_matrix[0]]
    else:
        tau_levels = [_['tutor_rule_tau'] for _ in args_matrix[0]]
    
    sim_idx = -1 if max_steps is None else max_steps
    is_converged = (lambda err: np.isfinite(err[sim_idx]) and np.isfinite(err[0]) and
                                err[sim_idx] < err[0])
    try:
        converged = np.asarray([[is_converged(_['error_trace']) for _ in crt_row]
                                for crt_row in res_matrix])
    except IndexError:
        converged = np.asarray([[is_converged(_) for _ in crt_row]
                                for crt_row in res_matrix])
    
    n_levels = len(tau_levels)

    main_ax = fig.add_subplot(1, 1, 1)
    
    if max_steps is not None:
        n_reps = max_steps
    else:
        try:
            n_reps = len(res_matrix[0][0]['error_trace'])
        except IndexError:
            n_reps = len(res_matrix[0][0])
    
    pos0 = main_ax.get_position()
    
    x0 = pos0.x0 + 0.01
    x1 = pos0.x1 - 0.01
    y0 = pos0.y0 + 0.01
    y1 = pos0.y1 - 0.01

    factorx = (x1 - x0)/n_levels
    edgex = factorx/20
    factory = (y1 - y0)/n_levels
    edgey = factory/20

    for i in xrange(n_levels):
        for j in xrange(n_levels):
            crt_ax = fig.add_axes([x0 + edgex + factorx*i, y0 + edgey + factory*(n_levels - 1 - j),
                                   factorx - 2*edgex, factory - 2*edgey],
                                  axisbg=[0, 0, 0, 0] if converged[j][i] else no_convergence_color)
            try:
                crt_ax.plot(res_matrix[j][i]['error_trace'], lw=1,
                  color=err_color)
            except IndexError:
                crt_ax.plot(res_matrix[j][i], lw=1, color=err_color)
            crt_ax.set_xlim(-5, n_reps)
            crt_ax.set_ylim(-3, max_error)

            crt_ax.spines['right'].set_color('none')
            crt_ax.spines['left'].set_color('none')
            crt_ax.spines['top'].set_color('none')
            crt_ax.spines['bottom'].set_color('none')

            crt_ax.xaxis.set_ticks([])
            crt_ax.yaxis.set_ticks([])
            
    ticks = range(n_levels)
    ticklabels = ['{:.0f}'.format(crt_tau) for crt_tau in tau_levels]
    
    main_ax.set_xticks(ticks)
    main_ax.set_yticks(ticks)
    
    main_ax.set_xticklabels(ticklabels, rotation=45, fontsize=8)
    main_ax.set_yticklabels(ticklabels, fontsize=8)
 
    main_ax.set_xlabel(r'$\tau_\mathrm{tutor}$', labelpad=xlabelpad)
    main_ax.set_ylabel(r'$\frac {\alpha \tau_1 - \beta \tau_2} {\alpha - \beta}$',
                       labelpad=ylabelpad)
    
    main_ax.set_xlim(-0.5, n_levels-0.5)
    main_ax.set_ylim(n_levels-0.5, -0.5)

def get_firing_rate(res, tmax):
    try:
        crt_times0 = np.asarray(res['student_spike'].t)
        N = res['student_spike'].N
    except KeyError:
        crt_times0 = np.asarray(res['student'].t)
        N = res['student'].N

    crt_times = crt_times0[crt_times0 < tmax]
    return len(crt_times)*1000.0/tmax/N

def make_convergence_movie(fname, res, target, fps=30.0, length=5.0, muscle=0, ymax=None,
                           invert=False, idxs=None, colorcycle=[[1.0, 0.2, 0.2], [0.2, 0.2, 1.0]],
                           labels=None):
    """ Make a movie showing convergence of motor program.
    
    Parameters
    ----------
      fname: string
          Output file name.
      res: list of dictionaries, or tuple of lists
          The results of the simulation. If this is a tuple of lists, a trace is drawn
          for each of the simulations.
      target: array
          Target motor program.
      fps: float
          Frame rate.
      length: float
          Length of movie (in seconds).
      muscle: None, int
          If not `None`, a muscle to focus on. If `None`, all muscles are
          shown.
      ymax: None, float
          If not `None`, sets the `ylim` to `(0, ymax)`.
      invert: bool
          Set to `True` to invert the colors.
      idxs: None, sequence of int
          If not `None`, use only a subset of all the frames in `res`.
      colorcycle: list of colors
          Colors to be used for drawing the results. These can be in any format
          understood by Matplotlib (letters, strings, triples or quadruples of float).
      labels: list of strings
          Labels to use for the motor programs. Set to `None` to disable the legend.
    """
    if hasattr(res[0], 'has_key'):
        res = [res]
        
    if idxs is None:
        idxs = range(min(len(x) for x in res))
#    motor_idxs = np.asarray([x.has_key('motor') for x in res]).nonzero()[0]
    res_with_motor = [np.asarray(x)[idxs] for x in res]
    
    nframes = int(fps*length)
    
    times = res_with_motor[0][0]['motor'].t[:len(target[0])]
    
    if invert:
        bgdict = {'facecolor': 'k'}
    else:
        bgdict = {}
        
    fig = plt.figure(figsize=(4, 3), **bgdict)
    
    plt.plot(times, target[muscle], c='w' if invert else 'k', lw=0.5)
    lines = []
    for i, crt_res in enumerate(res_with_motor):
        c_idx = (i % len(colorcycle))
        if labels is not None:
            extra = {'label': labels[i]}
        else:
            extra = {}
        line, = plt.plot(times, crt_res[0]['motor'].out[muscle][:len(target[0])],
                         c=colorcycle[c_idx], **extra)
        lines.append(line)

    title = plt.title('Stage 0', fontsize=16)
    if labels is not None:
        legend = plt.legend()
    else:
        legend = None
    if ymax is not None:
        plt.ylim((0, ymax))
    plt.xlabel('time', fontsize=12)
    plt.ylabel('motor output', fontsize=12)
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.gca()
    if invert:
        ax.set_axis_bgcolor('black')
        ax_color = [0.8, 0.8, 0.8]
        title.set_color(ax_color)
        for crt_spine in ax.spines.values():
            crt_spine.set_color(ax_color)
        ax.xaxis.label.set_color(ax_color)
        ax.yaxis.label.set_color(ax_color)
        ax.tick_params(axis='x', colors=ax_color)
        ax.tick_params(axis='y', colors=ax_color)
        
        if legend is not None:
            for crt_text in legend.get_texts():
                crt_text.set_color(ax_color)
        
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    
    def animate(i):
        """ Draw the `i`th frame. """
        r_idx = i*len(res_with_motor[0])/nframes
        # find an index where we have motor information
        while r_idx > 0 and not all(crt_res[r_idx].has_key('motor') for crt_res in res_with_motor):
            r_idx -= 1
            
        title.set_text('Stage {}'.format(idxs[r_idx]))
        for crt_line, crt_res in zip(lines, res_with_motor):
            crt_line.set_ydata(crt_res[r_idx]['motor'].out[muscle][:len(target[0])])
        return tuple(lines) + (title,)
    
    def init():
        for crt_line, crt_res in zip(lines, res_with_motor):
            crt_line.set_ydata(crt_res[0]['motor'].out[muscle])
        return tuple(lines) + (title,)
    
    anim = animation.FuncAnimation(fig, animate, np.arange(nframes), init_func=init, blit=True)
    anim.save(fname, dpi=300, fps=fps, savefig_kwargs=bgdict, bitrate=2048.0)
