
# helper for reading profiler files

# python -m cProfile -o profiler.txt ogmlrun.py

import pstats; p = pstats.Stats('profiler.txt'); p.strip_dirs().sort_stats('time').print_stats(40)

import pstats; p = pstats.Stats('profiler.txt'); p.strip_dirs().sort_stats('cumulative').print_stats(40)

