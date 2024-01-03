from irrep_actions.utils.module_attr_mixin import ModuleAttrMixin
from irrep_actions.utils.normalizer import LinearNormalizer


class BasePolicy(ModuleAttrMixin):
    def __init__(self, action_dim, num_action_steps, robot_state_len, world_state_len, z_dim):
        super().__init__()

        self.action_dim = action_dim
        self.num_action_steps = num_action_steps
        self.robot_state_len = robot_state_len
        self.world_state_len = world_state_len
        self.z_dim = z_dim

        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
