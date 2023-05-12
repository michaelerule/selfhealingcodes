# Self-healing codes: *How stable neural populations can track continually reconfiguring neural representations*

Rule, M. E., & O’Leary, T. (2022). Self-healing codes: How stable neural populations can track continually reconfiguring neural representations. Proceedings of the National Academy of Sciences, 119(7), e2106692119. [doi: https://doi.org/10.1073/pnas.2106692119.](https://doi.org/10.1073/pnas.2106692119) 

### Abstract

As an adaptive system, the brain must retain a faithful representation of the world while continuously integrating new information. Recent experiments have measured population activity in cortical and hippocampal circuits over many days and found that patterns of neural activity associated with fixed behavioral variables and percepts change dramatically over time. Such “representational drift” raises the question of how malleable population codes can interact coherently with stable long-term representations that are found in other circuits and with relatively rigid topographic mappings of peripheral sensory and motor signals. We explore how known plasticity mechanisms can allow single neurons to reliably read out an evolving population code without external error feedback. We find that interactions between Hebbian learning and single-cell homeostasis can exploit redundancy in a distributed population code to compensate for gradual changes in tuning. Recurrent feedback of partially stabilized readouts could allow a pool of readout cells to further correct inconsistencies introduced by representational drift. This shows how relatively simple, known mechanisms can stabilize neural tuning in the short term and provides a plausible explanation for how plastic neural codes remain integrated with consolidated, long-term representations.

### Significance

The brain is capable of adapting while maintaining stable long-term memories and learned skills. Recent experiments show that neural responses are highly plastic in some circuits, while other circuits maintain consistent responses over time, raising the question of how these circuits interact coherently. We show how simple, biologically motivated Hebbian and homeostatic mechanisms in single neurons can allow circuits with fixed responses to continuously track a plastic, changing representation without reference to an external learning signal.

### Repository Contents

This repository cotains example iPython simulation notebooks for *Self-healing codes: How stable neural populations can track continually reconfiguring neural representations.* Simulations are implemented in `master.py` and various figures have their own notebooks. 

 - `config.py`: Sets up python environment and defines helper routines
 - `master.py`: Simulation routines
 - `standard_options.py`: Parameter values used in manuscript
 - `f1bc.ipynb`: Figure 1b and 1c
 - `f1ef.ipynb`: Figure 1e and 1f
 - `f2.ipynb`: Figure 2
 - `f3.ipynb`: Figure 3
 - `s1.ipynb`: Supplemental figure 1
 - `s2.ipynb`: Supplemental figure 2
 - `s3.ipynb`: Supplemental figure 3
 - `s4.ipynb`: Supplemental figure 4
 - `s5.ipynb`: Supplemental figure 5
