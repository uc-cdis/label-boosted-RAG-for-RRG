from transformers import PreTrainedModel
from XREM_config import XREMConfig
from XREM_module import RETRIEVAL_MODULE


class XREM(PreTrainedModel):
    config_class = XREMConfig

    def __init__(self, config):
        super().__init__(config)
        self.cosine_sim_module = RETRIEVAL_MODULE(
            mode="cosine-sim",
            config=config.albef_retrieval_config,
            checkpoint=config.albef_retrieval_ckpt,
            topk=config.albef_retrieval_top_k,
            input_resolution=config.albef_itm_image_resolution,
            delimiter=config.albef_retrieval_delimiter,
            max_token_len=config.albef_retrieval_max_token_len,
        )

        self.itm_module = RETRIEVAL_MODULE(
            mode="image-text-matching",
            config=config.albef_itm_config,
            checkpoint=config.albef_itm_ckpt,
            topk=config.albef_itm_top_k,
            input_resolution=config.albef_itm_image_resolution,
            delimiter=config.albef_itm_delimiter,
            max_token_len=config.albef_itm_max_token_len,
        )

    def forward(self, reports, image_dataset_cosine, image_dataset_itm=None):
        output = self.cosine_sim_module.predict(image_dataset_cosine, reports)
        if self.config.albef_itm_top_k > 0:
            reports = [
                report.split(self.config.albef_retrieval_delimiter) for report in output
            ]
            output = self.itm_module.predict(image_dataset_itm, reports)
        return output
