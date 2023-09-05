from .attentions.scale_dot_product_attention import ScaleDotProductAttention
from .attentions.additive_attention import AdditiveAttention
from .attentions.position_aware_attention_scaling import PositionAwareAttentionScaling
from .attentions.sim_attention import SimAttention
from .attentions.soft_attention import SOFTAttention
from .attentions.linformer_attention import LinformerAttention
from .attentions.cosformer_attention import CosformerAttention
from .attentions.norm_attention import NormAttention
from .attentions.experiment import Experiment
from .attentions.diag_attention import DiagAttention
from .attentions.local_attention import LocalAttention
from .attentions.robust import RobustAttention
from .attentions.relu_value_attention import REVAttention


def get_attention_by_config(Config):
    """
    Factory function to get specific attention module by Config
    """
    if hasattr(Config, 'ATTENTION_TYPE'):
        attention_type = Config.ATTENTION_TYPE
    else:
        attention_type = 'dot_product'  # default scale dot product attention
    valid_a_types = ['dot_product', 'additive', 'paas',
                     'paas-linear', 'simal1', 'simal2', "soft", "linformer",
                     'cosformer', 'norm', 'experiment',
                     'diag', 'local', 'transnormer', 'robust',
                     'reva']
    if attention_type not in valid_a_types:
        raise ValueError(
            f"attention_type should be one of {valid_a_types}, but got {attention_type}")

    max_seq_length = Config.MAX_SEQ_LENGTH
    q_same_as_k = False
    d_tensor = Config.D_MODEL // Config.N_HEAD
    if attention_type == 'dot_product':
        attention = ScaleDotProductAttention()
    elif attention_type == 'additive':
        attention = AdditiveAttention(d_tensor)
    elif attention_type == 'paas':
        attention = PositionAwareAttentionScaling(max_seq_length)
    elif attention_type == 'paas-linear':
        attention = PositionAwareAttentionScaling(
            max_seq_length, wp_init='linear')
    elif attention_type == 'simal1':
        attention = SimAttention(use_l1_norm=True)
    elif attention_type == 'simal2':
        attention = SimAttention(use_l1_norm=False)
    elif attention_type == 'soft':
        q_same_as_k = True
        attention = SOFTAttention()
    elif attention_type == 'linformer':
        attention = LinformerAttention(
            max_seq_length, Config.LINFORMER_K)
    elif attention_type == 'cosformer':
        attention = CosformerAttention()
    elif attention_type == 'norm':
        attention = NormAttention(d_tensor, normalization=Config.NORM_ATTENTION_TYPE)
    elif attention_type == 'diag':
        # TODO: make only first half layers use diag attention
        attention = DiagAttention(Config.DIAG_BLOCK_SIZE)
    elif attention_type == 'experiment':
        attention = Experiment(max_seq_length, Config.DIAG_BLOCK_SIZE)
    elif attention_type == 'local':
        attention = LocalAttention(Config.LOCAL_ATTENTION_R)
    elif attention_type == 'robust':
        attention = RobustAttention(max_seq_length, Config.DIAG_BLOCK_SIZE)
    elif attention_type == 'reva':
        attention = REVAttention()

    return attention, q_same_as_k
