from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent

from agents.acmfql_bc_distill import ACFQLAgent as ACMFQLAgent_bc
from agents.acimfql import ACFQLAgent as ACIMFQLAgent
agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    acmfql_bc=ACMFQLAgent_bc,
    acimfql=ACIMFQLAgent,
)
