from fvf.model.common.module_attr_mixin import ModuleAttrMixin
from fvf.model.common.normalizer import LinearNormalizer

class BasePolicy(ModuleAttrMixin):
    def __init__(self, obs_dim, action_dim, num_obs_steps, num_action_steps):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_obs_steps = num_obs_steps
        self.num_action_steps = num_action_steps

        self.normalizer = LinearNormalizer()

    def reset(self):
        pass

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
