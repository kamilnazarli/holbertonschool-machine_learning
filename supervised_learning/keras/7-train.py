#!/usr/bin/env python3
'''module documented'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    '''
    function documented
    '''
    if validation_data is not None:
        early_stopping = K.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=patience)
        if learning_rate_decay:
            lr_schedule = K.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=alpha,
                decay_steps=epochs,
                decay_rate=decay_rate,
                staircase=True
            )
        return network.fit(x=data, y=labels,
                           batch_size=batch_size, epochs=epochs,
                           callbacks=[early_stopping],
                           verbose=verbose, shuffle=shuffle,
                           validation_data=validation_data)
    else:
        return network.fit(x=data, y=labels, batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose, shuffle=shuffle)
