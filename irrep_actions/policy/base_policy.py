from irrep_actions.utils.module_attr_mixin import ModuleAttrMixin
from irrep_actions.utils.normalizer import LinearNormalizer


class BasePolicy(ModuleAttrMixin):
    def __init__(self, obs_dim, action_dim, num_obs_steps, num_action_steps, horizon, z_dim):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_obs_steps = num_obs_steps
        self.num_action_steps = num_action_steps
        self.horizon = horizon
        self.z_dim = z_dim

        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
