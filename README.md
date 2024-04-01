# tinygill
 GILL with TinyLLama for Kandinsky 2.2

## Usage

`Train&Generation.ipynb` for interactive work, `strain.sh` for submitting jobs to Slurm in batch mode.
* `dataset.py` loads COCO captions dataset and Kandinsky2.2 CLIP-text encoder.
* `param_count.py` contains auxilary functions for tracking trainable parameters.
* `tiny_gill.py` TinyGILL model architecture.
* `train.py` train loop for one epoch.
* `generate.py` text2img generation with Kandinsky 2.2 and TinyGILL.
