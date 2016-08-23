#! /usr/bin/env python
""" Create job files and run jobs to see effects of mismatching timescales at
tutor and student.
"""

import sys
import os
import time

import numpy as np
import cPickle as pickle

try:
  from argparse import ArgumentParser
  from subprocess import check_output
except:
  sys.exit("Need Python version 2.7. Make sure the right module is loaded.")

if __name__ == '__main__':
  version = sys.version_info
  if version.major != 2 or version.minor < 7:
    sys.exit("Need Python version 2.7. Make sure the right module is loaded.")

  parser = ArgumentParser(description="Create and run job files for testing " +
      " the effect of timescale mismatch in the spiking simulations.")

  parser.add_argument('--defaults', help="pickle containing default parameters",
      required=True)
  parser.add_argument('--nreps', help="number of steps per simulation",
      default=1500)

  args = parser.parse_args()

  taus = 10*2**np.arange(8)

  with open(args.defaults, 'rb') as inp:
    defaults = pickle.load(inp)

  tau1, tau2 = defaults['plasticity_taus']

  for (i, tau_student) in enumerate(taus):
    alpha = float(tau_student - tau2)/(tau1 - tau2)
    beta = alpha - 1

    run_name = "job/job_bird_tscale_{}.sh".format(i)
    job_name = "bird_t_{}".format(i)

    job_script = """#!/bin/bash
#!/bin/bash
#PBS -q production
#PBS -N {job_name}
#PBS -l select=1:ncpus=1
#PBS -l walltime=24:00:00
#PBS -l place=free
#PBS -V

date
cd $PBS_O_WORKDIR

SCRIPT_PATH=songbird/batch
SCRIPT_NAME=run_once.py
SCRIPT_FULL="$SCRIPT_PATH/$SCRIPT_NAME"
""".format(job_name=job_name)

    for (j, tau_tutor) in enumerate(taus):
      base_name = "songspike_tscale_batch_{}.{}".format(i, j) + "." + \
        time.strftime("%y%m%d.%H%M")

      # start setting up the command line options
      cmdline_opts = ("--plasticity_params=\"({},{})\" --tutor_rule_tau={}".
          format(alpha, beta, tau_tutor))
      cmdline_opts += " --nreps={} --defaults={}".format(args.nreps,
          args.defaults)

      log_name = "log/" + base_name + "_out.txt"

      cmdline_opts += " --out=out/" + base_name + ".pkl"
      cmdline_opts += " --log=" + log_name

      job_name = "bird_t_{}.{}".format(i, j)

      job_script += "$SCRIPT_FULL {cmdline_opts} 2>&1\n\n".format(
          cmdline_opts=cmdline_opts)

    job_script += "\ndate\n"

    with open(run_name, 'wt') as out:
      out.write(job_script)

    # and now submit them
    qsub_opts = run_name
    print("About to call qsub " + qsub_opts + "...")
    out = check_output(['qsub'] + qsub_opts.split())
    print("Answer: " + out)
