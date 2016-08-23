#! /usr/bin/env python
""" Figure out the best learning parameters for the reinforcement-based
simulations. """

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

  parser = ArgumentParser(description="Create and run job files for finding" +
      " good learning rates for reinforcement simulations.")

  parser.add_argument('--defaults', help="pickle containing default parameters",
      required=True)
  parser.add_argument('--nreps', help="number of steps per simulation",
      default=6000)
  parser.add_argument('--n_per_job', help="number of simulations per job",
      default=3)

  args = parser.parse_args()

  plasticity_rates = np.arange(0.01, 0.12, 0.01)*1e-8
  tutor_rates = np.arange(0.003, 0.012, 0.001)

  time_tag = time.strftime("%y%m%d.%H%M")

  all_rates = [(ip, it) for ip in xrange(len(plasticity_rates))
                        for it in xrange(len(tutor_rates))]

  n_jobs = int(np.ceil(float(len(all_rates))/args.n_per_job))
  crt_rate_idx = 0

  for i in xrange(n_jobs):
    crt_rates = all_rates[crt_rate_idx:crt_rate_idx+args.n_per_job]

    run_name = "job/job_bird_reinf_optim_{}.sh".format(i)
    job_name = "reinf_opt{}".format(i)

    job_script = """#!/bin/bash
#!/bin/bash
#PBS -q production
#PBS -N {job_name}
#PBS -l select=1:ncpus=1
#PBS -l walltime=36:00:00
#PBS -l place=free
#PBS -V

date
cd $PBS_O_WORKDIR

SCRIPT_PATH=songbird/batch
SCRIPT_NAME=run_once.py
SCRIPT_FULL="$SCRIPT_PATH/$SCRIPT_NAME"
""".format(job_name=job_name)

    for j in xrange(len(crt_rates)):
      ip, it = crt_rates[j]
      plasticity_rate = plasticity_rates[ip]
      tutor_rate = tutor_rates[it]
      base_name = "song_reinf_optim_batch_{}.{}".format(ip, it) + "." + time_tag

      # start setting up the command line options
      cmdline_opts =  "--tutor_tau_out=40.0 --tutor_rule_type=reinforcement"
      cmdline_opts += " --plasticity_learning_rate={}".format(plasticity_rate)
      cmdline_opts += " --tutor_rule_learning_rate={}".format(tutor_rate)
      cmdline_opts += " --relaxation=200.0 --relaxation_conductor=200.0"
      cmdline_opts += " --tutor_rule_relaxation=0 --tutor_rule_tau=0"
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

    crt_rate_idx += args.n_per_job
