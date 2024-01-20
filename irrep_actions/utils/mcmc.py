import torch

def iterative_dfo(
    policy,
    obs,
    actions,
    action_dist,
    temp=1.0,
    num_iterations=3,
    iteration_std=0.33,
):
    """ DFO MCMC  """
    device = obs.device
    B = actions.size(0)
    num_actions = actions.size(1)
    zero = torch.tensor(0, device=device)
    resample_std = torch.tensor(iteration_std, device=device)

    for i in range(num_iterations):
        logits = policy.forward(obs, actions)
        log_probs = logits / temp
        probs = torch.softmax(logits, dim=-1)

        if i < (num_iterations - 1):
            idxs = torch.multinomial(probs, num_actions, replacement=True)
            actions = actions[torch.arange(B).unsqueeze(-1), idxs]
            actions += torch.normal(
                zero, resample_std, size=actions.shape, device=device
            )
            actions = torch.clamp(actions, action_dist[0], action_dist[1])
            resample_std *= 0.5

    idxs = torch.multinomial(probs, num_samples=1, replacement=True)
    actions = actions[torch.arange(B).unsqueeze(-1), idxs].squeeze(1)

    return probs, actions

def langevin_actions(
    policy,
    obs,
    actions,
    action_dist,
    num_iterations=5,
    sampler_stepsize_init=1e-1,
    sampler_stepsize_decay=0.8,
    noise_scale=1.0,
    grad_clip=None,
    delta_action_clip=0.1,
    apply_exp=True,
    use_polynomial_rate=True,
    sampler_stepsize_final=1e-5,
    sampler_stepsize_power=2.0,
):
    """ Langevin MCMC  """
    B = actions.size(0)
    stepsize = sampler_stepsize_init

    # Init scheduler
    if use_polynomial_rate:
        scheduler = PolynomialScheduler(
            sampler_stepsize_init,
            sampler_stepsize_final,
            sampler_stepsize_power,
            num_iterations
        )
    else:
        scheduler = ExponentialScheduler(
            sampler_stepsize_init,
            sampler_stepesize_decay
        )

    # Langevin step updates
    zero = torch.tensor(0, device=policy.device)
    noise_scale = torch.tensor(noise_scale, device=policy.device)
    for step in range(num_iterations):
        langevin_lambda = 1.0

        de_dact, energy = gradient_wrt_action(
            policy,
            obs,
            actions,
            apply_exp
        )

        if grad_clip is not None:
            de_dact = torch.clamp(de_dact, -grad_clip, grad_clip)

        gradient_scale = 0.5
        de_dact = gradient_scale * langevin_lambda * de_dact
        de_dact += torch.normal(
            zero, langevin_lambda * noise_scale, size=de_dact.shape, device=de_dact.device
        )
        delta_actions = stepsize * de_dact
        delta_actions = torch.clamp(delta_actions, -delta_action_clip, delta_action_clip)

        actions = actions - delta_actions
        actions = torch.clamp(actions, action_dist[0], action_dist[1])

        stepsize = scheduler.get_rate(step + 1)

    actions = actions.detach()
    logits = policy.forward(obs, actions)
    probs = torch.softmax(logits, dim=-1)
    idxs = torch.multinomial(probs, num_samples=1, replacement=True)
    actions = actions[torch.arange(B).unsqueeze(-1), idxs].squeeze(1)

    return probs, actions

def gradient_wrt_action(policy, obs, actions, apply_exp):
    actions.requires_grad_()
    energy = policy.forward(obs, actions)

    if apply_exp:
        energy = torch.exp(energy)

    # Get energy gradient wrt action
    de_dact = torch.autograd.grad(energy.sum(), actions, create_graph=True)[0]

    return de_dact.detach(), energy

class PolynomialScheduler(object):
    def __init__(self, init, final, power, num_steps):
        self.init = init
        self.final = final
        self.power = power
        self.num_steps = num_steps

    def get_rate(self, step):
        return ((self.init - self.final) *
                ((1 - (float(step) / float(self.num_steps-1))) ** (self.power))
               ) + self.final

class ExponentialScheduler(object):
    def __init__(self, init, decay):
        self.rate = init
        self.decay = decay

    def get_rate(self, step):
        self.rate *= self.decay
        return self.rate
