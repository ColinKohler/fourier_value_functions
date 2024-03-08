import torch
import math
from imitation_learning.utils import harmonics

def iterative_dfo(
    policy,
    obs,
    actions,
    action_dist,
    harmonic_actions=False,
    normalizer=None,
    lmax=None,
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
        if harmonic_actions:
            mag = actions[:,:,:,0].unsqueeze(3)
            theta = normalizer["action"].unnormalize(actions)[:,:,0,1]
            logits = policy(obs, mag, theta)
        else:
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
    use_polynomial_rate=True,
    sampler_stepsize_final=1e-5,
    sampler_stepsize_power=2.0,
    harmonic_actions=False,
    normalizer=None,
    lmax=None,
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
    zero = torch.tensor(0, device=obs.device)
    noise_scale = torch.tensor(noise_scale, device=obs.device)
    for step in range(num_iterations):
        langevin_lambda = 1.0

        de_dact, energy = gradient_wrt_action(
            policy,
            obs,
            actions,
            harmonic_actions=harmonic_actions,
            normalizer=normalizer,
            lmax=lmax
        )

        if grad_clip is not None:
            de_dact = torch.clamp(de_dact, -grad_clip, grad_clip)

        gradient_scale = 0.5
        noise = torch.normal(
            zero, langevin_lambda * noise_scale, size=de_dact.shape, device=de_dact.device
        )
        #delta_actions = stepsize * de_dact + math.sqrt(2 * stepsize) * noise
        delta_actions = stepsize * (gradient_scale * langevin_lambda * de_dact + noise)
        delta_actions = torch.clamp(delta_actions, -delta_action_clip, delta_action_clip)

        actions = actions - delta_actions
        actions = torch.clamp(actions, action_dist[0], action_dist[1])
        actions = actions.detach()

        stepsize = scheduler.get_rate(step + 1)

    if harmonic_actions:
        W = policy.forward(obs, actions[:,:,:,0].unsqueeze(3))
        theta = normalizer['action'].unnormalize(actions)[:,:,0,1]
        logits = harmonics.get_energy(W.view(-1, W.size(2)), theta.view(-1, 1), lmax).view(obs.size(0), -1)
    else:
        logits = policy.forward(obs, actions)

    probs = torch.softmax(logits, dim=-1)

    return probs, actions

def gradient_wrt_action(policy, obs, actions, harmonic_actions=False, normalizer=None, lmax=None):
    #assert not torch.is_grad_enabled()

    #if harmonic_actions:
    #    W = policy.forward(obs, actions[:,:,:,0].unsqueeze(3))
    #    theta = normalizer['action'].unnormalize(actions)[:,:,0,1]
    #    energy = harmonics.get_energy(W.view(-1, W.size(2)), theta.view(-1, 1), lmax).view(obs.size(0), -1)
    #else:
    energy = policy.forward(obs, actions)
    def Ex_sum(actions):
        energy = policy.forward(obs, actions)
        return energy.sum()

    # Get energy gradient wrt action
    with torch.set_grad_enabled(True):
        #de_dact1 = torch.autograd.grad(energy.sum(), actions, create_graph=True)[0]
        de_dact = torch.autograd.functional.jacobian(Ex_sum, actions) * -1.0

    return de_dact, energy

def compute_grad_norm(de_dact):
    return torch.linalg.norm(de_dact, dim=1, ord=float('inf'))

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
