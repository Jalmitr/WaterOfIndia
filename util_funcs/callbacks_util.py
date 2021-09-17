
import tensorflow
from datetime import datetime


def callbacks_util(img_height, img_width, csv_log=True, ely_stop=True, lr_schd=False, mdl_ckpt=True, rlrplat=True):
    all_callbacks = []
    dt_text = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")

    if csv_log is True:
        log_file = "./woi_model/unet_{n1}x{n2}.log".format(n1=img_height, n2=img_width)  # + dt_text
        csvlogging = tensorflow.keras.callbacks.CSVLogger(log_file, separator=",", append=True)
        all_callbacks.append(csvlogging)
    else:
        temp = 0

    if ely_stop is True:
        early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=4, verbose=1,
                                                                  mode="min", baseline=None, restore_best_weights=True)
        all_callbacks.append(early_stopping)
    else:
        temp = 0

    if lr_schd is True:
        # not used right now, maybe needed later
        all_callbacks.append()
    else:
        temp = 0

    if mdl_ckpt is True:
        ckpt_file = "./woi_model/unet_{n1}x{n2}".format(n1=img_height, n2=img_width)
        model_checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(
            ckpt_file, monitor="val_loss", verbose=0, save_best_only=True, save_weights_only=False, mode="min")
        all_callbacks.append(model_checkpoint)
    else:
        temp = 0

    if rlrplat is True:
        rlrp = tensorflow.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, verbose=1, mode="min", min_delta=0.0001, cooldown=0, min_lr=1e-5)
        all_callbacks.append(rlrp)
    else:
        temp = 0

    return all_callbacks


def save_model_explicit(unet, img_height, img_width, save_flag=False):

    if save_flag is True:
        # dt_text = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
        model_save_path = "./woi_model/unet_{n1}x{n2}_explicit".format(n1=img_height, n2=img_width)  # + dt_text
        tensorflow.keras.models.save_model(unet, model_save_path, overwrite=True, include_optimizer=True,
                                           save_format="t5", signatures=None, options=None, save_traces=True)
    else:
        print("Model not saved explicitly. Verify the directory if saved with checkpoint callback.")
    return 0
