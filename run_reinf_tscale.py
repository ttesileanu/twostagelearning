#! /usr/bin/env python
""" Create job files and run jobs to see effects of mismatching timescales at
tutor and student in the reinforcement simulations.
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
      " the effect of timescale mismatch in the reinforcement simulations.")

  parser.add_argument('--defaults', help="pickle containing default parameters",
      required=True)
  parser.add_argument('--nreps', help="number of steps per simulation",
      default=8000)
  parser.add_argument('--n_per_job', help="number of simulations per job",
      default=3)
  parser.add_argument('--plasticity_rate', help="plasticity rule learning rate",
      default=6e-10)
  parser.add_argument('--tutor_rate', help="tutor rule learning rate",
      default=0.006)

  args = parser.parse_args()

  taus = 10*2**np.arange(8)
  n_taus = len(taus)

  with open(args.defaults, 'rb') as inp:
    defaults = pickle.load(inp)

  tau1, tau2 = defaults['plasticity_taus']
  time_tag = time.strftime("%y%m%d.%H%M")

  all_idxs = [(i1, i2) for i1 in xrange(n_taus) for i2 in xrange(n_taus)]

  n_jobs = int(np.ceil(float(len(all_idxs))/args.n_per_job))
  crt_idx = 0

  for i in xrange(n_jobs):
    crt_idxs = all_idxs[crt_idx:crt_idx+args.n_per_job]

    run_name = "job/job_bird_reinf_tscale_{}.sh".format(i)
    job_name = "reinf_tsc{}".format(i)

    job_script = """#!/bin/bash
#!/bin/bash
#PBS -q production
#PBS -N {job_name}
#PBS -l select=1:ncpus=1
#PBS -l walltime=48:00:00
#PBS -l place=free
#PBS -V

date
cd $PBS_O_WORKDIR

SCRIPT_PATH=songbird/batch
SCRIPT_NAME=run_once.py
SCRIPT_FULL="$SCRIPT_PATH/$SCRIPT_NAME"
""".format(job_name=job_name)

    for j in xrange(len(crt_idxs)):
      i1, i2 = crt_idxs[j]
      tau_student = taus[i1]
      tau_tutor = taus[i2]

      alpha = float(tau_student - tau2)/(tau1 - tau2)
      beta = alpha - 1

      base_name = "song_reinf_tscale_batch_{}.{}".format(i1, i2)+"."+time_tag

      # start setting up the command line options
      cmdline_opts =  "--tutor_tau_out=40.0 --tutor_rule_type=reinforcement"
      cmdline_opts += " --plasticity_params=\"({},{})\"".format(alpha, beta)
      cmdline_opts += " --tutor_rule_tau={}".format(tau_tutor)
      cmdline_opts += " --plasticity_learning_rate={}".format(
          args.plasticity_rate)
      cmdline_opts += " --tutor_rule_learning_rate={}".format(
          args.tutor_rate)
      cmdline_opts += " --relaxation=200.0 --relaxation_conductor=200.0"
      cmdline_opts += " --tutor_rule_relaxation=0"
      cmdline_opts += " --tutor_rule_compress_rates=True"
      cmdline_opts += " --nreps={} --defaults={}".format(args.nreps,
          args.defaults)

      log_name = "log/" + base_name + "_out.txt"

      cmdline_opts += " --out=out/" + base_name + ".pkl"
      cmdline_opts += " --log=" + log_name

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

    crt_idx += args.n_per_job
