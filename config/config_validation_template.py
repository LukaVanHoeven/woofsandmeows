"""
DISCLAIMER: 
This code was previously part of Joris Heemskerks Bachelors thesis, 
and is being re-used here. All rights are reserved to Joris Heemskerk, 
and Technolution BV, Gouda NL. Joris was granted the rights to use and 
modify this code, at the express notion that a disclaimer was put in.
"""

""" Config yaml template

    This template can be used to validate the composition of the configuration yaml file.
    Minimal viable yaml file looks like (_'s are placeholders for data):

    ```yaml
    general:
        num_data_workers: _
    jobs:
        job0:
            sample_rate: _
            stride: _
            duration: _
            n_fft: _
            hop_length: _
            n_mels: _
            top_db: _
            train_val_split: _
            batch_size: _

            model: _
            model_params: _
            optimiser: _
            b: _
            learning_rate: _
            weight_decay: _
            n_epochs: _
            k_folds: _

            tune: _
            n_trials: _
            n_startup_trials: _
    ```
"""

CONFIG_TEMPLATE = {
    'type': 'object',
    'properties': {
        'general': {
            'type': 'object',
            'properties': {
                'num_data_workers': {
                    'type': 'number',
                    'minimum': 0
                },
            },
            'required': [
                'num_data_workers',
            ],
            'additionalProperties' : False
        },
        'jobs': {
            'type': 'object',
            'patternProperties': {
                '^job\\d+$': {
                    'type': 'object',
                    'properties': {
                        'sample_rate': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1
                        },
                        'stride': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1
                        },
                        'duration': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1
                        },
                        'n_fft': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1
                        },
                        'hop_length': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1
                        },
                        'n_mels': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1
                        },
                        'top_db': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1
                        },
                        'train_val_split': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 2,
                            'maxItems': 2
                        },
                        'batch_size': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1
                        },

                        'model': {
                            'type': 'string', 
                        },
                        'model_params': {
                            'type': 'object', 
                        },
                        'optimiser': {
                            'type': 'string', 
                        },
                        'b': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1
                        },
                        'learning_rate': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1
                        },
                        'weight_decay': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1
                        },
                        'n_epochs': {
                            'type': 'number', 
                            'minimum': 1
                        },
                        'k_folds': {
                            'type': 'number', 
                            'minimum': 1
                        },
                        
                        'tune': {
                            'type': 'boolean', 
                        },
                        'n_trials': {
                            'type': 'number', 
                        },
                        'n_startup_trials': {
                            'type': 'number', 
                        },
                    },
                    'required': [
                        'sample_rate',
                        'stride',
                        'duration',
                        'n_fft',
                        'hop_length',
                        'n_mels',
                        'top_db',
                        'train_val_split',
                        'batch_size',

                        'model',
                        'model_params',
                        'optimiser',
                        'b',
                        'learning_rate',
                        'weight_decay',
                        'n_epochs',
                        'k_folds',
                    ],
                    'additionalProperties' : False
                }
            }
        },
    },
    'required': ['general', 'jobs'],
    'additionalProperties' : False
}
