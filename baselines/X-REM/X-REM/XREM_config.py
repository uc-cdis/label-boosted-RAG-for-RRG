from transformers import PretrainedConfig


class XREMConfig(PretrainedConfig):
    model_type = "XREM"

    def __init__(
        self,
        albef_retrieval_config="configs/Cosine-Retrieval.yaml",
        albef_retrieval_ckpt="../../jaehwan_CXR_ReFusE/ALBEF/output/sample/pretrain/checkpoint_59.pth",
        albef_retrieval_top_k=50,
        albef_retrieval_delimiter="[SEP]",
        albef_retrieval_image_resolution=256,
        albef_retrieval_max_token_len=25,
        albef_itm_config="configs/ITM.yaml",
        albef_itm_ckpt="../../jaehwan_CXR_ReFusE/ALBEF/output/sample/ve/checkpoint_7.pth",
        albef_itm_top_k=10,
        albef_itm_delimiter="[SEP]",
        albef_itm_image_resolution=384,
        albef_itm_max_token_len=30,
        **kwargs
    ):
        self.albef_retrieval_config = albef_retrieval_config
        self.albef_retrieval_ckpt = albef_retrieval_ckpt
        self.albef_retrieval_top_k = albef_retrieval_top_k
        self.albef_retrieval_delimiter = albef_retrieval_delimiter
        self.albef_retrieval_image_resolution = albef_retrieval_image_resolution
        self.albef_retrieval_max_token_len = albef_retrieval_max_token_len

        self.albef_itm_config = albef_itm_config
        self.albef_itm_ckpt = albef_itm_ckpt
        self.albef_itm_top_k = albef_itm_top_k
        self.albef_itm_delimiter = albef_itm_delimiter
        self.albef_itm_image_resolution = albef_itm_image_resolution
        self.albef_itm_max_token_len = albef_itm_max_token_len

        super().__init__(**kwargs)
