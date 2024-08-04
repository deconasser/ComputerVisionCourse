import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) # disable absl INFO and WARNING log messages

import gc
import shutil
import numpy as np
import matplotlib.pyplot as plt 
from utils.dataset import DatasetController
from utils.visualize import save_plot

from utils.general import set_seed
from utils.general import yaml_save
from utils.general import increment_path

from utils.option import parse_opt
from utils.option import update_opt
from utils.option import model_dict

from utils.metrics import metric_dict
from utils.rich2polars import table_to_df
# from utils.activations import get_custom_activations

from rich import box as rbox
from rich.table import Table
from rich.console import Console
from rich.terminal_theme import MONOKAI

def main(opt):
    """ Get the save directory for this run """
    save_dir = str(increment_path(Path(opt.project) / opt.name, overwrite=opt.overwrite, mkdir=True))

    """ Path to save configs """
    path_configs = Path(save_dir, 'configs')
    path_configs.mkdir(parents=True, exist_ok=True)

    """ Set seed """
    opt.seed = set_seed(opt.seed)

    """ Add custom function """
    # get_custom_activations()

    """ Save init options """
    yaml_save(path_configs / 'opt.yaml', vars(opt))

    """ Update options """
    opt = update_opt(opt)
    shuffle = False

    """ Save updated options """
    yaml_save(path_configs / 'updated_opt.yaml', vars(opt))

    """ Preprocessing dataset """
    dataset = DatasetController(configsPath=opt.dataConfigs,
                                resample=opt.resample,
                                # startTimeId=opt.startTimeId,
                                splitRatio=(opt.trainsz, opt.valsz, 1-opt.trainsz-opt.valsz),
                                workers=opt.workers,
                                lag=opt.lag, 
                                ahead=opt.ahead, 
                                offset=opt.offset,
                                savePath=save_dir,
                                polarsFilling=opt.polarsFilling,
                                machineFilling=opt.machineFilling,
                                low_memory=opt.low_memory,
                                normalization=opt.normalization).execute(cyclicalPattern=opt.cyclicalPattern)
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.GetData(shuffle=shuffle)
    scaler = dataset.scaler

    del dataset
    gc.collect()

    """ Create result table """
    console = Console(record=True)
    table = Table(title="[cyan]Results", show_header=True, header_style="bold magenta", box=rbox.ROUNDED, show_lines=True)
    [table.add_column(f'[green]{name}', justify='center') for name in ['Name', 'Time', *list(metric_dict.keys())]]

    """ Train models """
    # from models.test import build_Baseline_idea1, build_Baseline_idea2, build_Baseline_idea4, build_Baseline_ave, build_Baseline_5ave, build_Bi_LSTSM
    # from models.test import build_Bi_LSTSM
    # from utils.metrics import score
    # for f in [build_Baseline_idea1, build_Baseline_idea2, build_Baseline_idea4, build_Baseline_ave, build_Baseline_5ave]:
    #     model = f(opt.lag, 1)
    #     yhat = model.predict(X_test[:, :, 1:2])
    #     scores = score(y=y_test, 
    #                      yhat=yhat, 
    #                      r=4)
    #     table.add_row(f.__name__, '0s', *scores)
    #     console.print(table)
    # import time
    # from keras.optimizers import Adam
    # from keras.losses import MeanSquaredError
    # from tensorflow.keras.utils import Sequence 
    # from keras.callbacks import EarlyStopping
    # from keras.callbacks import ReduceLROnPlateau
    # from utils.general import convert_seconds
    # class DataGenerator(Sequence):
    #     def __init__(self, x, y, batchsz):
    #         self.x, self.y = x, y
    #         self.batch_size = batchsz

    #     def __len__(self):
    #         return int(np.ceil(len(self.x) / float(self.batch_size)))

    #     def __getitem__(self, idx):
    #         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    #         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
    #         return batch_x, batch_y
    # model = build_Bi_LSTSM(seed=opt.seed, input_len=opt.lag, predict_len=opt.ahead, chanels=2)
    # start = time.time()
    # model.compile(optimizer=Adam(learning_rate=opt.lr), loss=MeanSquaredError())
    # min_delta = 0.001
    # history = model.fit(DataGenerator(x=X_train, y=y_train, batchsz=opt.batchsz), 
    #                               validation_data=DataGenerator(x=X_val, y=y_val, batchsz=opt.batchsz),
    #                               epochs=opt.epochs,
    #                               callbacks=[EarlyStopping(monitor='val_loss', patience=opt.patience, min_delta=min_delta),
    #                                          ReduceLROnPlateau(monitor='val_loss',
    #                                                            factor=0.1,
    #                                                            patience=opt.patience / 5,
    #                                                            verbose=0,
    #                                                            mode='auto',
    #                                                            min_delta=min_delta * 10,
    #                                                            cooldown=0,
    #                                                            min_lr=0)])
    # time_used = convert_seconds(time.time() - start)
    # yhat = model.predict(X_test)
    # scores = score(y=y_test, 
    #                  yhat=yhat, 
    #                  r=4)
    # table.add_row('BiLSTM', time_used, *scores)
    # console.print(table)

    for item in model_dict:
        if not vars(opt)[f'{item["model"].__name__}']: continue
        shutil.copyfile(item['config'], path_configs/os.path.basename(item['config']))
        datum = train(model=item['model'], 
                      modelConfigs=item['config'], 
                      data=[[X_train, y_train], [X_val, y_val], [X_test, y_test]], 
                      save_dir=save_dir,
                      ahead=opt.ahead, 
                      seed=opt.seed, 
                      normalize_layer=None,
                      learning_rate=opt.lr,
                      epochs=opt.epochs, 
                      patience=opt.patience,
                      optimizer=opt.optimizer, 
                      loss=opt.loss,
                      batchsz=opt.batchsz,
                      r=opt.round,
                      enc_in=1,
                      scaler=scaler)
        table.add_row(*datum)
        console.print(table)
        console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)  
    table_to_df(table).write_csv(os.path.join(save_dir, 'results.csv'))

def train(model, modelConfigs, data, save_dir, ahead,
          seed: int = 941, 
          normalize_layer=None,
          learning_rate: float = 1e-3,
          epochs: int = 10_000_000, 
          patience: int = 1_000,
          optimizer:str = 'Adam', 
          loss:str = 'MSE',
          batchsz:int = 64,
          r: int = 4,
          enc_in: int = 1,
          scaler = None) -> list:
    # import tensorflow as tf
    # model = tf.keras.models.load_model('VanillaLSTM__Tensorflow')
    # model.summary()

    model = model(input_shape=data[0][0].shape[-2:],
                  modelConfigs=modelConfigs, 
                  output_shape=ahead, 
                  seed=seed,
                  normalize_layer=None,
                  save_dir=save_dir,
                  enc_in=enc_in)
    model.build()
    # model.model.built = True
    # model.load('LTSF_Linear__Tensorflow_bestckpt.index')
    model.fit(patience=patience, 
              optimizer=optimizer, 
              loss=loss, 
              epochs=epochs, 
              learning_rate=learning_rate, 
              batchsz=batchsz,
              X_train=data[0][0], y_train=data[0][1],
              X_val=data[1][0], y_val=data[1][1])
    model.save(file_name=f'{model.__class__.__name__}')
    

    weight=os.path.join(save_dir, 'weights', f"{model.__class__.__name__}_best.h5")
    if not os.path.exists(weight): weight = model.save(file_name=model.__class__.__name__)
    else: model.save(save_dir=save_dir, file_name=model.__class__.__name__)
    # weight = r'runs\exp809\weights\VanillaLSTM__Tensorflow_best.h5'
    if weight is not None: model.load(weight)

    # predict values
    # yhat = model.predict(X=data[2][0][:10_000], name='test')
    # ytrainhat = model.predict(X=data[0][0][:10_000], name='train')
    # yvalhat = model.predict(X=data[1][0][:10_000], name='val')
    yhat = model.predict(X=data[2][0], scaler=scaler)
    ytrainhat = model.predict(X=data[0][0], scaler=scaler)
    yvalhat = model.predict(X=data[1][0], scaler=scaler)

    # calculate scores
    # print(data[2][1], yhat)
    # print(yhat[:10])
    # print(data[2][1][:10])
    scores = model.score(y=data[2][1], 
                         yhat=yhat, 
                         # path=save_dir,
                         r=r)

    # plot values
    model.plot(save_dir=save_dir, y=data[0][1], yhat=ytrainhat, dataset='Train')
    model.plot(save_dir=save_dir, y=data[1][1], yhat=yvalhat, dataset='Val')
    model.plot(save_dir=save_dir, y=data[2][1], yhat=yhat, dataset='Test')

    model.plot(save_dir=save_dir, y=data[0][1][:100], yhat=ytrainhat[:100], dataset='Train100')
    model.plot(save_dir=save_dir, y=data[1][1][:100], yhat=yvalhat[:100], dataset='Val100')
    model.plot(save_dir=save_dir, y=data[2][1][:100], yhat=yhat[:100], dataset='Test100')

    return [model.__class__.__name__, model.time_used, *scores]

def run(**kwargs):
    """ 
    Usage (example)
        import train
        train.run(all=True, 
                  configsPath=data.yaml,
                  lag=5,
                  ahead=1,
                  offset=1)
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt(ROOT=ROOT)
    main(opt)