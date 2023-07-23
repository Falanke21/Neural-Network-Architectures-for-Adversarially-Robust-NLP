from .attentions.scale_dot_product_attention import ScaleDotProductAttention
from .attentions.additive_attention import AdditiveAttention
from .attentions.position_aware_attention_scaling import PositionAwareAttentionScaling
from .attentions.sim_attention import SimAttention
from .attentions.soft_attention import SOFTAttention
from .attentions.linformer_attention import LinformerAttention


def get_attention_by_config(Config):
    """
    Factory function to get specific attention module by Config
    """
    if hasattr(Config, 'ATTENTION_TYPE'):
        attention_type = Config.ATTENTION_TYPE
    else:
        attention_type = 'dot_product'  # default scale dot product attention
    valid_a_types = ['dot_product', 'additive', 'paas',
                     'paas-linear', 'simal1', 'simal2', "soft", "linformer"]
    if attention_type not in valid_a_types:
        raise ValueError(
            f"attention_type should be one of {valid_a_types}, but got {attention_type}")

    max_seq_length = Config.MAX_SEQ_LENGTH
    q_same_as_k = False
    if attention_type == 'dot_product':
        attention = ScaleDotProductAttention()
    elif attention_type == 'additive':
        d_tensor = Config.D_MODEL // Config.N_HEAD
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

    return attention, q_same_as_k
