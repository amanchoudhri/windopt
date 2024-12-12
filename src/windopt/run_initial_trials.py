"""
Queue 
"""

from winc3d.les import start_les
from gch import gch

if __name__ == "__main__":
    # run 100 GCH trials
    gch(n_trials=100)
    # run 12 LES trials
    les(n_trials=12)
