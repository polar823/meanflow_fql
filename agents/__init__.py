from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent

from agents.acmfql_bc_distill import ACFQLAgent as ACMFQLAgent_bc
from agents.acimfql import ACFQLAgent as ACIMFQLAgent
from agents.steer_flow_policy_with_latent_space import SFPLSAgent
agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    acmfql_bc=ACMFQLAgent_bc,
    acimfql=ACIMFQLAgent,
    sfpls = SFPLSAgent,
)
