#!/usr/bin/env python3
'''module documented'''
import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    '''
    function documented
    '''
    callbacks = []
    if validation_data is not None and early_stopping:
        early_stopping = K.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=patience)
        callbacks.append(early_stopping)
    if validation_data is not None and learning_rate_decay:
        K.backend.set_value(network.optimizer.learning_rate,
                            alpha)

        def decay_lr(epoch):
            return alpha / (1 + decay_rate * epoch)
        lr_callback = K.callbacks.LearningRateScheduler(decay_lr,
                                                        verbose=1)
        callbacks.append(lr_callback)
    if validation_data is not None and save_best:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        )
        callbacks.append(checkpoint)
    return network.fit(x=data, y=labels,
                       batch_size=batch_size, epochs=epochs,
                       callbacks=callbacks, verbose=verbose,
                       shuffle=shuffle, validation_data=validation_data)
