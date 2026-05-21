# Pre-alignment

UniReg supports SAMCoarse pre-alignment through pre-computed `coarse_phi` files. 
The training and evaluation pipeline reads these files through the unified dataset loader when `use_pre_align: true`. 
