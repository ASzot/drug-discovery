

def get_default_args():
    params = {
        # Standard definition of epsilon
        'eps': 1e-5,

        # Hyperparams of the PPO equation
        'clip_param': 0.2,

        # Number of update epochs
        'n_epoch': 4,

        # Number of mini batches to use in updating
        'n_mini_batch': 32,

        # Coefficients for the loss term. (all relative to the action loss which is 1.0)
        'value_coeff': 0.5,
        'entropy_coeff': 0.01,

        # Learning rate of the optimizer
        'lr': 7e-4,

        # Clip gradient norm
        'max_grad_norm': 0.5,

        # Number of steps to generate actions
        'n_steps': 5,

        # Total number of frames to train on
        'n_frames': 10e6,

        # Should we use GPU?
        'cuda': True,

        # Discounted reward factor
        'gamma': 0.99,

        # Number of environments to run in paralell this is like the batch size
        'n_envs': 16,

        'save_interval': 500,
        'log_interval': 10,

        'model_dir': 'weights',
    }

    return params



