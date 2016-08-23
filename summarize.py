#! /usr/bin/env python
""" Summarize results files by keeping only the 'average_error' trace.

This finds a template that matches all the files provided on the command line,
and returns them in an array of the appropriate size.

The template must have the form
  <string1><idx1><string2><idx2>...<stringN><idxN><stringN+1>
where <idx1>, ..., <idxN> are integers ranging from 0 to some maximum value kN.
An error is issued if the files do not follow this convention.
"""

import sys
import os
import copy

import numpy as np
import cPickle as pickle

if __name__ == '__main__':
  version = sys.version_info
  try:
    if version.major != 2 or version.minor < 7:
      sys.exit("Need Python version 2.7. Make sure the right module is loaded.")
  except:
    sys.exit("Need Python version 2.7. Make sure the right module is loaded.")

  filenames = copy.copy(sys.argv[1:])

  # sequentially eat away from the left of the strings
  template = []

  while True:
    common = os.path.commonprefix(filenames)
    if len(common) > 0:
      template.append(common)
      if all(len(_) == len(common) for _ in filenames):
        # we've finished the templat
        break
      # check that for every file, this is followed by an integer
      # and find out its maximum value
      num_max = 0
      for i, crt_name in enumerate(filenames):
        crt_name_left = crt_name[len(common):]
        # adding the space to make sure at least one character is not a digit,
        # so that `index` doesn't fail
        num_len = [_.isdigit() for _ in (crt_name_left + ' ')].index(False)
        if num_len == 0:
          raise Exception('Command line arguments do not obey template.')

        num = int(crt_name_left[:num_len])
        if num > num_max:
          num_max = num

        filenames[i] = crt_name_left[num_len:]

      template.append(num_max+1)
    else:
      if max(len(_) for _ in filenames) > 0:
        raise Exception('Command line arguments do not obey template.')
      else:
        break

  print('Processing ' + ''.join(_ if isinstance(_, (str, unicode)) else
      '<idx'+str(_)+'>' for _ in template) + '...')

  dims = template[1::2]
  res_array = np.empty(dims, dtype='object')
  args_array = np.empty(dims, dtype='object')

  # a sneaky way to help iterate over all the files
  all_idxs = np.asarray(np.ones(dims, dtype=bool).nonzero()).T

  for crt_idxs in all_idxs:
    sys.stdout.write('.')
    sys.stdout.flush()
    crt_name = ''.join(_ if i%2 == 0 else str(crt_idxs[(i-1)/2])
        for (i, _) in enumerate(template))
    with open(crt_name, 'rb') as inp:
      crt_res = pickle.load(inp)

    args_array[tuple(crt_idxs)] = crt_res['args']
    res_array[tuple(crt_idxs)] = np.asarray([_['average_error']
        for _ in crt_res['res']])

  sys.stdout.write('\n')

  common_args = copy.copy(args_array.ravel()[0])
  # find common arguments
  for crt_args in args_array.ravel():
    to_delete = []
    for key in common_args:
      if not crt_args.has_key(key):
        # if the key doesn't appear in some argument lists, it can't be common
        to_delete.append(key)
      else:
        # check whether the value is the same as in the common list
        crt_value = crt_args[key]
        common_value = common_args[key]

        if not np.array_equal(crt_value, common_value):
          to_delete.append(key)
    
    for key in to_delete:
      common_args.pop(key)

  # keep only the differing arguments in args_array
  for crt_args in args_array.ravel():
    for key in common_args:
      crt_args.pop(key, None)

  # finally, store the result
  out_name0 = (''.join(_ if isinstance(_, (str, unicode)) else str(_)
      for _ in template))
  out_root = os.path.splitext(out_name0)[0]
  out_name = out_root + '_summary.pkl'

  print('Writing results to ' + out_name + '.')

  with open(out_name, 'wb') as out:
    pickle.dump({'common_args': common_args,
                  'args_array': args_array,
                  'res_array': res_array}, out, 2)
