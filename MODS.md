Now the synthetic tensor generator supports MLM-like tensors if a flag is passed during init (tensor_type = 'seq').
The memcpy measurements in that case are to be executed outside the iterator, during the training script instead, as a deliberate choice to make the treatment uniform with measurements on other data pipelines — this is the major modification in behaviour. Apart from that, it works the same.
For the same reason, the iter method just returns the mini batch (as a dict con input ids, attention mask and labels), they are to be moved to device manually in the training loop.

Other, smaller fixes are performed on the harness.py script; the most important is printing a warning message in the clear_cache function if the clearing is not successful, since on systems without root access the command would not work. However, given that using harness.py always runs the synthetic job before the cold run, the cache should be free of the needed dataset regardless in practical applications.

Finally, we introduce direct GPU occupancy monitoring as an available feature in the DataStallProfiler class, using nvidia-smi.
