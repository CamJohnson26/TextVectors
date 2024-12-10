# OOM errors
Run `nvidia-smi, see if ollama or something else is still open and using GPU RAM`
model.half() was the key, sped things up a lot