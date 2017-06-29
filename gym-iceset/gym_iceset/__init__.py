import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register (
    id = 'IceTrainerEnv-v0',
    entry_point='gym_iceset.envs:IceTrainerEnv',
    kwargs={'dataset_path': '/Users/kv/workspace/research/rlloop/gym-iceset/data/'}
    )

register (
    id = 'LoopCreatorEnv-v0',
    entry_point='gym_iceset.envs:LoopCreatorEnv',
    kwargs={'dataset_path': '/Users/kv/workspace/research/rlloop/gym-iceset/data/'}
    )