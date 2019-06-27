This file documents the corresponding training notebook and its training details for each optimized model.

Updated on 04-10-2019

--------------------------------------------------------------------------------

synth_newcrap_001_unet_mse.ipynb (combo 1 w/o new neuron movies)
- model: synth_newcrap_001_unet_mse.7
- image size: 128*128
- image number:
    - train set: 8529 items
    - valid set: 1849 items
- image type: combo of neuron, mito, MISC images from server
- resnet18, unet_learner, fit_one_cycle, mse_loss
- Training pipeline:
    - lr = 1e-3
    - size = 128
          do_fit(f'{nb_name}.0', lr, cycle_len=50)
          learn.unfreeze()
          do_fit(f'{nb_name}.1', slice(1e-5,lr), cycle_len=10)
    - size = 256
          learn = learn.load(f'{nb_name}.1')
          do_fit(f'{nb_name}.2', lr/100, cycle_len=50)
          learn.unfreeze()
          do_fit(f'{nb_name}.3', slice(1e-5,lr/10), cycle_len=10)
    - size = 512
          learn = learn.load(f'{nb_name}.3')
          do_fit(f'{nb_name}.4', lr/100, cycle_len=50)
          learn.unfreeze()
          do_fit(f'{nb_name}.5', slice(1e-6,lr/100), cycle_len=10)
    - size = 1024
          learn = learn.load(f'{nb_name}.5')
          do_fit(f'{nb_name}.6', lr/100, cycle_len=10)
          learn.unfreeze()
          do_fit(f'{nb_name}.7', slice(1e-6,lr/100), cycle_len=10)

synth_newcrap_002_unet_mse.ipynb (combo 2 w/ new neuron movies)
- model: synth_newcrap_002_unet_mse.7
- image size: 128*128
- image number:
    - train set: 7846 items
    - valid set: 2210 items
- image type: combo of neuron, mito, MISC images from server, microtubule
- resnet18, unet_learner, fit_one_cycle, mse_loss, saveBestModel
- Training pipeline:
    - lr = 1e-3
    - size = 128
          do_fit(f'{nb_name}.0', lr, cycle_len=100)
          learn.unfreeze()
          do_fit(f'{nb_name}.1', slice(1e-5,lr), cycle_len=50)
    - size = 256
          learn = learn.load(f'{nb_name}.1')
          do_fit(f'{nb_name}.2', lr/100, cycle_len=100)
          learn.unfreeze()
          do_fit(f'{nb_name}.3', slice(1e-5,lr/10), cycle_len=50)
    - size = 512
          learn = learn.load(f'{nb_name}.3')
          do_fit(f'{nb_name}.4', lr/100, cycle_len=100)
          learn.unfreeze()
          do_fit(f'{nb_name}.5', slice(1e-6,lr/100), cycle_len=50)
    - size = 1024
          learn = learn.load(f'{nb_name}.5')
          do_fit(f'{nb_name}.6', lr/100, cycle_len=100)
          learn.unfreeze()
          do_fit(f'{nb_name}.7', slice(1e-6,lr/100), cycle_len=50)

transfer_learning_neuron_002_unet_mse.ipynb
- model: transfer_learning_neuron_002_unet_mse.7
- image size: 128*128
- image number:
    - train set: 2400 items
    - valid set: 1410 items
- train/valid set type: more neuron czi files
- resnet18, unet_learner, fit_one_cycle, mse_loss, saveBestModel
- transfer learning
    - Base network: synth_newcrap_001_unet_mse.7 (final model trained on combo of neuron, mito and server images)
    - Training pipeline:
        - lr = 5e4
        - size = 128
              learn = learn.load('synth_newcrap_001_unet_mse.7') #transfer learning
              do_fit(f'{nb_name}.0', lr, cycle_len=100)
              learn.unfreeze()
              do_fit(f'{nb_name}.1', slice(1e-5,lr), cycle_len=100)
        - size = 256
              learn = learn.load(f'{nb_name}.1')
              do_fit(f'{nb_name}.2', lr/100, cycle_len=100)
              learn.unfreeze()
              do_fit(f'{nb_name}.3', slice(1e-5,lr/10), cycle_len=100)
        - size = 512
              learn = learn.load(f'{nb_name}.3')
              do_fit(f'{nb_name}.4', lr/100, cycle_len=100)
              learn.unfreeze()
              do_fit(f'{nb_name}.5', slice(1e-6,lr/100), cycle_len=50)
        - size = 1024
              learn = learn.load(f'{nb_name}.5')
              do_fit(f'{nb_name}.6', lr/100, cycle_len=10)
              learn.unfreeze()
              do_fit(f'{nb_name}.7', slice(1e-6,lr/100), cycle_len=10)              

transfer_learning_neuron_003_unet_mse_oneshot.ipynb
- model: transfer_learning_neuron_003_unet_mse_oneshot.7
- image size: 128*128
- image number:
    - train set: 1 items
    - valid set: 0 items
- train/valid set type: one neuron tif slice
- resnet18, unet_learner, fit_one_cycle, mse_loss
- transfer learning
    - Base network: synth_newcrap_001_unet_mse.7 (final model trained on combo of neuron, mito and server images)
    - Training pipeline:
        - lr = 5e4
        - size = 128
              learn = learn.load('synth_newcrap_001_unet_mse.7') #transfer learning
              do_fit(f'{nb_name}.0', lr, cycle_len=100)
              learn.unfreeze()
              do_fit(f'{nb_name}.1', slice(1e-5,lr), cycle_len=100)
        - size = 256
              learn = learn.load(f'{nb_name}.1')
              do_fit(f'{nb_name}.2', lr/100, cycle_len=100)
              learn.unfreeze()
              do_fit(f'{nb_name}.3', slice(1e-5,lr/10), cycle_len=100)
        - size = 512
              learn = learn.load(f'{nb_name}.3')
              do_fit(f'{nb_name}.4', lr/100, cycle_len=100)
              learn.unfreeze()
              do_fit(f'{nb_name}.5', slice(1e-6,lr/100), cycle_len=100)
        - size = 1024
              learn = learn.load(f'{nb_name}.5')
              do_fit(f'{nb_name}.6', lr/100, cycle_len=100)
              learn.unfreeze()
              do_fit(f'{nb_name}.7', slice(1e-6,lr/100), cycle_len=100) 
           