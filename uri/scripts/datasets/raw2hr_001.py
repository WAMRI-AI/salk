from pathlib import Path
import shutil
import random
import os
import superres.helpers as helpers
from fastprogress import progress_bar
import PIL

from pathlib import Path
import shutil
import random
import os
from fastprogress import progress_bar

data_path = Path('/scratch/bpho')
sources = data_path/'datasources'
raw = sources/'Processed_and_Raw_Movie'
processed_2d = sources/'Processed_and_Raw_Movie/2D SR Airyscan process'
processed = sources/'Processed_and_Raw_Movie/Airyscan process'



datasets = data_path/'datasets'
raw2hr = datasets/'raw2hr_001'

