from losses import binary_crossentropy, dice, focal_loss, jaccard_loss
from losses import composite_loss
from callbacks import KerasTerminateOnMetricNaN
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, EarlyStopping
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.callbacks import ReduceLROnPlateau


def get_loss_function(framework, loss_name, loss_params):
    """Get the training loss function from the config dictionary."""
    if loss_name == 'binary_crossentropy':
        return binary_crossentropy(framework)
    elif loss_name == 'dice':
        return dice(framework)
    elif loss_name == 'focal_loss':
        return focal_loss(framework, loss_params)
    elif loss_name == 'jaccard':
        return jaccard_loss(framework)
    elif loss_name == 'composite_loss':
        return composite_loss(framework,
                              loss_a=get_loss_function(framework,
                                                       loss_params['loss_a'],
                                                       loss_params),
                              loss_b=get_loss_function(framework,
                                                       loss_params['loss_b'],
                                                       loss_params),
                              alpha=loss_params['alpha'],
                              **loss_params.pop('alpha'))


def get_callbacks(framework, callback_dict):
    """Get the callbacks needed for training in the required framework."""
    if framework == 'keras':
        callback_list = []

        if 'model_checkpoint' in callback_dict:
            model_checkpoint = ModelCheckpoint(
                filepath=callback_dict['model_checkpoint'].get('filepath',
                                                               'm.hdf5'),
                monitor=callback_dict['model_checkpoint'].get('monitor',
                                                              'val_loss'),
                verbose=callback_dict['model_checkpoint'].get('verbose', 0),
                save_best_only=callback_dict['model_checkpoint'].get(
                    'save_best_only', False),
                save_weights_only=callback_dict['model_checkpoint'].get(
                    'save_weights_only', False),
                mode=callback_dict['model_checkpoint'].get('mode', 'auto'),
                period=callback_dict['model_checkpoint'].get('period', 1)
                )
            callback_list.append(model_checkpoint)

        if 'early_stopping' in callback_dict:
            early_stopping = EarlyStopping(
                monitor=callback_dict['early_stopping'].get('monitor',
                                                            'val_loss'),
                min_delta=callback_dict['early_stopping'].get('min_delta', 0),
                patience=callback_dict['early_stopping'].get('patience', 0),
                verbose=callback_dict['early_stopping'].get('verbose', 0),
                mode=callback_dict['early_stopping'].get('mode', 'auto'),
                baseline=callback_dict['early_stopping'].get('baseline', None),
                restore_best_weights=callback_dict['early_stopping'].get(
                    'restore_best_weights', False)
            )
            callback_list.append(early_stopping)
        if 'learning_rate_scheduler' in callback_dict:
            lr_scheduler = LearningRateScheduler(
                schedule=callback_dict['learning_rate_scheduler']['schedule'],
                verbose=callback_dict['learning_rate_scheduler'].get('verbose',
                                                                     0)
            )
            callback_list.append(lr_scheduler)

        if 'tensorboard' in callback_dict:
            tensorboard = TensorBoard(
                log_dir=callback_dict['tensorboard'].get('log_dir', './logs'),
                histogram_freq=callback_dict['tensorboard'].get(
                    'histogram_freq', 0),
                batch_size=callback_dict['tensorboard'].get('batch_size', 32),
                write_graph=callback_dict['tensorboard'].get('write_graph',
                                                             True),
                write_grads=callback_dict['tensorboard'].get('write_grads',
                                                             False),
                write_images=callback_dict['tensorboard'].get('write_images',
                                                              False),
                embeddings_freq=callback_dict['tensorboard'].get(
                    'embeddings_freq', 0),
                embeddings_layer_names=callback_dict['tensorboard'].get(
                    'embeddings_layer_names', None),
                embeddings_metadata=callback_dict['tensorboard'].get(
                    'embeddings_metadata', None),
                embeddings_data=callback_dict['tensorboard'].get(
                    'embeddings_data', None),
                update_freq=callback_dict['tensorboard'].get('update_freq',
                                                             'epoch')
            )
            callback_list.append(tensorboard)

        if 'reduce_lr_on_plateau' in callback_dict:
            reduce_lr_on_plateau = ReduceLROnPlateau(
                monitor=callback_dict['reduce_lr_on_plateau'].get('monitor',
                                                                  'val_loss'),
                factor=callback_dict['reduce_lr_on_plateau'].get('factor',
                                                                 0.1),
                patience=callback_dict['reduce_lr_on_plateau'].get('patience',
                                                                   10),
                verbose=callback_dict['reduce_lr_on_plateau'].get('verbose',
                                                                  0),
                mode=callback_dict['reduce_lr_on_plateau'].get('mode', 'auto'),
                min_delta=callback_dict['reduce_lr_on_plateau'].get(
                    'min_delta', 0.0001),
                cooldown=callback_dict['reduce_lr_on_plateau'].get('cooldown',
                                                                   0),
                min_lr=callback_dict['reduce_lr_on_plateau'].get('min_lr', 0)
            )
            callback_list.append(reduce_lr_on_plateau)

        if 'terminate_on_nan' in callback_dict:
            callback_list.append(TerminateOnNaN())

        if 'terminate_on_metric_nan' in callback_dict:
            terminate_on_metric_nan = KerasTerminateOnMetricNaN(
                metric=callback_dict['terminate_on_metric_nan'].get('metric',
                                                                    None)
            )
            callback_list.append(terminate_on_metric_nan)

        return callback_list

    # TODO: IMPLEMENT FOR OTHER FRAMEWORKS
